# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch.nn.functional as F
import torch

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.custom.metrics import TrainingMetrics


@register_criterion('candidate_penalty_cross_entropy')
class CandidatePenaltyCrossEntropyCriterion(FairseqCriterion):
    """Applies a (1-p(x_nt)) loss to each negative target ('candidate') x_nt."""

    def __init__(self, args, task):
        super().__init__(args, task)
        self.rank_alpha = args.rank_alpha
        self.candidate_type = args.candidate_type

    def forward(self, model, sample, reduce=True, compute_custom_metrics=True):
        net_output = model(**sample['net_input'])
        target = model.get_targets(sample, net_output)
        nsentences = target.size(0)
        target = target.view(-1)

        # -- mle loss
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        true_token_lprobs = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='none',
        )
        mle_loss = true_token_lprobs.sum()

        # -- custom loss
        # Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))

        # - form negative targets
        with torch.no_grad():
            # E.g. DABCC | D | EFFGD => {A,B,C} are negative targets.
            if self.candidate_type == 'prev_context':
                # Make 'the triangle'.
                ctx_cands = target.unsqueeze(0).expand(target.size(0), target.size(0))
                ctx_cands_ = (ctx_cands.tril(-1) + self.padding_idx)
                ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
                ctx_cands = ctx_cands.tril(-1) + ctx_cands_

                # Don't include the target for that timestep as a negative target.
                ctx_cands = ctx_cands.masked_fill(ctx_cands == target.unsqueeze(1), self.padding_idx)
                negative_targets = torch.zeros_like(lprobs).scatter_(1, ctx_cands, 1)
            else:
                raise NotImplementedError('candidate type %s' % self.candidate_type)

        # - compute loss
        one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)

        custom_loss = -torch.log(one_minus_probs)*negative_targets
        custom_loss = custom_loss.sum()

        loss = mle_loss + self.rank_alpha * custom_loss

        # -- metrics
        logits = net_output[0].view(-1, net_output[0].size(-1))
        true_token_logits = -F.nll_loss(
            logits,
            target,
            ignore_index=self.padding_idx,
            reduction='none',
        )

        orig = utils.strip_pad(target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = sample['target'].size(0) if self.args.sentence_avg else ntokens

        logging_output = {
            'custom_loss': utils.item(custom_loss.data),
            'loss': utils.item(mle_loss.data),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if compute_custom_metrics:
            custom_output = TrainingMetrics.ranking_metrics(logits, true_token_logits, sample, ntokens, target)
            for k, v in custom_output.items():
                logging_output[k] = v

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        custom_loss_sum = sum(log.get('custom_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'custom_loss': custom_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        from fairseq.custom.metrics import TrainingMetrics
        custom_output = TrainingMetrics.aggregate_and_normalize(logging_outputs)
        for k, v in custom_output.items():
            agg_output[k] = v

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2) if ntokens > 0 else 0.
        return agg_output
