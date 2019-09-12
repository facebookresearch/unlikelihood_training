# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch.nn.functional as F

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.custom.metrics import TrainingMetrics


@register_criterion('cross_entropy_wcustom_metrics')
class CrossEntropyCriterionWCustomMetrics(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True, compute_custom_metrics=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        logits = net_output[0].view(-1, net_output[0].size(-1))
        target = model.get_targets(sample, net_output)
        target = target.view(-1)
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        true_token_logits = -F.nll_loss(
            logits,
            target,
            ignore_index=self.padding_idx,
            reduction='none',
        )
        orig = utils.strip_pad(target, self.padding_idx)
        ntokens = orig.numel()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if compute_custom_metrics:
            custom_output = TrainingMetrics.ranking_metrics(logits, true_token_logits, sample, ntokens, target)
            for k, v in custom_output.items():
                logging_output[k] = v
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        from fairseq.custom.metrics import TrainingMetrics
        custom_output = TrainingMetrics.aggregate_and_normalize(logging_outputs)
        for k, v in custom_output.items():
            agg_output[k] = v

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum /ntokens / math.log(2) if ntokens > 0 else 0.
        return agg_output
