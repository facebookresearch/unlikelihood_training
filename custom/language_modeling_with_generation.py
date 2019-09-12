# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq.custom.sequence_penalty_loss import SequencePenaltyCriterion
import torch


@register_task('language_modeling_with_generation')
class LanguageModelingWithGenerationTask(LanguageModelingTask):
    """
    Train a language model, with generation-based evaluation.
    See `LanguageModelingTask` for args.
    """
    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary=output_dictionary, targets=targets)
        self._train_step = 1
        self._compute_metrics_interval = args.compute_metrics_interval
        self._sequence_level_train_rate = args.sequence_level_train_rate

        if self._sequence_level_train_rate > 0.0:
            self.sequence_criterion = self.build_sequence_criterion(args)
        self.generator = self.build_generator(args)

    def build_sequence_criterion(self, args):
        return SequencePenaltyCriterion(args, self)

    def start_epoch(self):
        self._train_step = 1

    @staticmethod
    def add_args(parser):
        parser.add_argument('--compute-metrics-interval', type=int, default=250,
                            help='compute custom metrics in the criterion once every `compute-metrics-interval` batches')
        parser.add_argument('--sequence-level-train-rate', type=float, default=0.0,
                            help='proportion of training steps to perform sequence level training')
        # - candidate_penalty
        parser.add_argument('--candidate-type', choices=['prev_context'],
                      default='prev_context')
        parser.add_argument('--rank-alpha', type=float)
        # - sequence
        parser.add_argument('--sequence-ngram-n', type=int, default=1)
        parser.add_argument('--sequence-prefix-length', type=int, default=16)
        parser.add_argument('--sequence-completion-length', type=int, default=48)
        parser.add_argument('--sequence-candidate-type', choices=['repeat', 'random'], default='repeat')
        parser.add_argument('--mask-p', type=float, default=0.5)

        # fmt: off
        LanguageModelingTask.add_args(parser)

    def build_generator(self, args):
        from fairseq.custom.sequence_generator import SequenceGenerator
        return SequenceGenerator(
            self.target_dictionary,
            temperature=1.0,
        )

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        agg_output = criterion.__class__.aggregate_logging_outputs(logging_outputs)
        if self._sequence_level_train_rate > 0.0:
            seq_agg_output = self.sequence_criterion.__class__.aggregate_logging_outputs(logging_outputs)
            for k, v in seq_agg_output.items():
                agg_output[k] = v
        return agg_output

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()

        do_mle_step=True
        # -- sequence level training
        if torch.rand(1).item() < self._sequence_level_train_rate:
            # check if current minibatch has at least one legal prefix, if not, do CE loss
            if sample['net_input']['src_tokens'].size(1) >= self.sequence_criterion.sequence_prefix_length:
                do_mle_step = False

                loss, sample_size, logging_output = self.sequence_criterion(model, sample,
                                                                            generator=self.generator)
                if ignore_grad:
                    loss *= 0
                optimizer.backward(loss)

        # -- normal training
        if do_mle_step:
            compute_custom_metrics = self._train_step % self._compute_metrics_interval == 0
            loss, sample_size, logging_output = criterion(model, sample, compute_custom_metrics=compute_custom_metrics)
            if ignore_grad:
                loss *= 0
            optimizer.backward(loss)

            # only track this for normal training steps, since sequence training always computes it own metrics.
            self._train_step += 1
        return loss, sample_size, logging_output

    def train_step_with_counts(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        compute_custom_metrics = self._train_step % self._compute_metrics_interval == 0
        loss, sample_size, logging_output = criterion(model, sample, compute_custom_metrics=compute_custom_metrics)
        assert 'best_tokens' in logging_output, 'best greedy tokens should be returned'
        best_tokens = logging_output['best_tokens']
        logging_output.pop('best_tokens')
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        self._train_step += 1
        return loss, sample_size, logging_output, best_tokens
