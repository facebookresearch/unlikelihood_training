# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch

from collections import defaultdict
from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.custom.evaluate_utils import batch_input_sequence_by_prefix_length
from fairseq.custom.metrics import ngram_metrics


@register_criterion('sequence_penalty')
class SequencePenaltyCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.sequence_ngram_n = args.sequence_ngram_n
        self.sequence_prefix_length = args.sequence_prefix_length
        self.sequence_completion_length = args.sequence_completion_length
        self.sequence_candidate_type = args.sequence_candidate_type
        self.mask_p = args.mask_p

    def forward(self, model, sample, reduce=True, generator=None):
        seq_len = sample['net_input']['src_tokens'].size(1)

        # make total number of tokens equal to the sequence length (for memory purposes)
        n_batches = seq_len // (self.sequence_prefix_length + self.sequence_completion_length)
        batch = batch_input_sequence_by_prefix_length(sample['net_input']['src_tokens'],
                                                      prefix_length=self.sequence_prefix_length)
        batch = batch[:n_batches]

        pred_toks, lprobs = generator.generate_completion_greedy_training(model, batch,
                                                                          completion_length=self.sequence_completion_length)
        if self.sequence_candidate_type == 'repeat':
            mask = ngram_repeat_mask(pred_toks, self.sequence_ngram_n).type_as(lprobs)
        elif self.sequence_candidate_type == 'random':
            mask = torch.bernoulli(torch.zeros_like(pred_toks, dtype=torch.float).fill_(self.mask_p))

        pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
        one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
        loss = -torch.log(one_minus_probs)*mask
        loss = loss.sum()

        ntokens = pred_toks.numel()  # number of output tokens (tokens in completions)
        nsentences = batch.size(0)
        sample_size = ntokens
        logging_output = {
            'seq_loss': utils.item(loss.data),
            'seq_ntokens': ntokens,
            'seq_nsentences': nsentences,
            'seq_repeat_mask': utils.item(mask.sum().data),
            'seq_sample_size': sample_size,
        }

        # Sum each statistic, which will be normalized by the number of sentences in `aggregate_logging_outputs`.
        stats = defaultdict(float)
        for tok_list in pred_toks.cpu().tolist():
            ms = ngram_metrics(tok_list)
            for k, v in ms.items():
                stats[k] += v
        for k, v in stats.items():
            logging_output[k] = v

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('seq_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('seq_ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('seq_nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('seq_sample_size', 0) for log in logging_outputs)
        repeat_mask = sum(log.get('seq_repeat_mask', 0) for log in logging_outputs)

        agg_output = {
            'seq_loss': loss_sum / max(sample_size, 1.0) / math.log(2),
            'seq_ntokens': ntokens,
            'seq_nsentences': nsentences,
            'seq_sample_size': sample_size,
            'seq_repeat_mask': repeat_mask / sample_size if sample_size > 0 else 0
        }

        for n in range(1, 5):
            key = 'pct_repeat_%dgrams' % n
            ngram_repeats = sum(log.get(key, 0) for log in logging_outputs)
            # Normalize by the number of sentences since this is the sum of per-sentence metrics.
            agg_output['seq_' + key] = ngram_repeats /nsentences if nsentences > 0 else 0

        if sample_size != ntokens:
            agg_output['seq_nll_loss'] = loss_sum / max(ntokens, 1.0) / math.log(2)
        return agg_output


def ngram_repeat_mask(xs, n):
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x)-n):
            ng = tuple(xl[j:j+n])
            if ng in seen:
                mask[i, j:j+n] = 1
            seen.add(ng)
    return mask

