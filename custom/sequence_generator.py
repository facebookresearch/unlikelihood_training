# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch
import torch.nn.functional as F


class SequenceGenerator(object):
    def __init__(self, tgt_dict, temperature=1.):
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.temperature = temperature

    def generate_completion_greedy_training(self, model, prefix_tokens, completion_length):
        model.eval()
        pred_toks = []
        context = prefix_tokens
        states = {}
        all_lprobs = []

        # First go over the context.
        for context_step in range(1, context.size(1)):
            _context = context[:, :context_step]
            _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)

        for tstep in range(completion_length):
            lprobs, attn_t = self._forward_one(model, context, incremental_states=states)
            pred_tok = lprobs.argmax(dim=1, keepdim=True)
            pred_toks.append(pred_tok)
            context = torch.cat((context, pred_tok), 1)
            all_lprobs.append(lprobs)

        pred_toks = torch.cat(pred_toks, 1)
        all_lprobs = torch.stack(all_lprobs, 1)
        return pred_toks, all_lprobs

    @torch.no_grad()
    def generate_completion(self, model, prefix_tokens, completion_length, topk, topp):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        pred_toks = []
        context = prefix_tokens
        states = {}

        # First go over the context.
        for context_step in range(1, context.size(1)):
            _context = context[:, :context_step]
            _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)

        for tstep in range(completion_length):
            logits, attn_t = self._forward_one(model, context, incremental_states=states, return_logits=True)
            pred_tok = self._topk_decode(logits, topk, topp)
            pred_toks.append(pred_tok)
            context = torch.cat((context, pred_tok), 1)
        pred_toks = torch.cat(pred_toks, 1)
        return pred_toks

    def _sample_topp(self, probs, sampling_topp):
        """Sample among the smallest set of elements whose cumulative probability mass exceeds p.
        See `"The Curious Case of Neural Text Degeneration"
        (Holtzman et al., 2019) <https://arxiv.org/abs/1904.09751>`_.
        Args:
            probs: (bsz x input_beam_size x vocab_size)  IK: here we dont have beam ! so bsz x vocab_size
                the model's log-probabilities over the vocabulary at the current step
        Return: A tuple of (trimed_probs, truncated_indices) where:
            trimed_probs: (bsz x input_beam_size x ?)
                the model's probabilities over the elements selected to sample from. The
                width of the third dimension is determined by top-P.
            truncated_indices: (bsz x input_beam_size x ?)
                the indices of the chosen elements.
        """
        # sort the last dimension (vocab dimension) in descending order
        sorted_probs, sorted_indices = probs.sort(descending=True)

        # compute a mask to indicate the words to be included in the top-P set.
        cumsum_probs = sorted_probs.cumsum(dim=1)
        mask = cumsum_probs.lt(sampling_topp)

        # note that mask was computed by 'lt'. One more word needs to be included
        # so that the cumulative probability mass can exceed p.
        cumsum_mask = mask.cumsum(dim=1)
        last_included = cumsum_mask[:, -1:]
        last_included.clamp_(0, mask.size()[1] - 1)
        mask = mask.scatter_(1, last_included, 1)

        # truncate unnecessary dims.
        max_dim = last_included.max()
        truncated_mask = mask[:, :max_dim + 1]
        truncated_probs = sorted_probs[:, :max_dim + 1]
        truncated_indices = sorted_indices[:, :max_dim + 1]

        # trim the words that are not in top-P by setting their probabilities
        # to 0, so that they would not be sampled later.
        trim_mask = 1 - truncated_mask
        trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)
        return trimed_probs, truncated_indices

    def _topk_decode(self, logits, topk, topp):
        """WARNING!!! This can modify the `self.pad` position of `logits`."""
        if topk == 1 and topp == 0:  # greedy
            logits[:, self.pad] = -math.inf  # as in fairseq code
            pred_tok = logits.argmax(dim=1, keepdim=True)

        else:
            if topk > 1:
                logits[:, self.pad] = -1e10  # never select pad
                logits = top_k_logits(logits, topk)
                pred_tok = torch.softmax(logits, -1).multinomial(1)
            else:
                assert topp > 0.0
                filtered_probs, bookkeep_idx = self._sample_topp(torch.softmax(logits, 1), sampling_topp=topp)
                selected = filtered_probs.multinomial(1)
                pred_tok = torch.gather(bookkeep_idx, index=selected, dim=1)

        return pred_tok

    def _forward_one(self, model, tokens, incremental_states=None, temperature=1., return_attn=False, return_logits=False, **decoder_kwargs):
        if incremental_states is not None:
            decoder_out = list(model.decoder(tokens, None, incremental_state=incremental_states, return_attn=return_attn, **decoder_kwargs))
        else:
            decoder_out = list(model.decoder(tokens, None, return_attn=return_attn, **decoder_kwargs))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn['attn']
        if attn is not None:
            if type(attn) is dict:
                attn = attn['attn']
            attn = attn[:, :, -1, :]  # B x L x t
        if return_logits:
            logits_t = decoder_out[0][:, -1, :]
            return logits_t, attn
        log_probs = model.get_normalized_probs(decoder_out, log_probs=True)
        log_probs = log_probs[:, -1, :]
        return log_probs, attn


def top_k_logits(logits, k):
    """Masks everything but the k top entries as -infinity (1e10).
    From: https://github.com/huggingface/pytorch-pretrained-BERT"""
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e5, logits)