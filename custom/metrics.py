# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict, Counter
from fairseq.custom.sequence_generator import SequenceGenerator
from fairseq.custom.sequence_generator import top_k_logits

from fairseq import utils
from nltk import ngrams


class TrainingMetrics(object):
    REPEAT_CONTEXT_LENGTHS = [16, 32, 128, 512]
    METRIC_NAMES = ['target_rank', 'median_target_rank',
                    'hits_at_1', 'hits_at_10']
    for l in REPEAT_CONTEXT_LENGTHS:
        METRIC_NAMES.extend([
            'repeat_at_1/%d' % l,
            'wrong_repeat_at_1/%d' % l,
            'human_repeat_at_1/%d' % l])

    @staticmethod
    def ranking_metrics(logits, true_token_logits, sample, ntokens, targets, topk=1, topp=0.0):
        """Compute summed metrics on a batch."""
        negative_targets = (logits > true_token_logits[:, None]).float()
        negative_targets_count = negative_targets.sum(dim=1)

        target_rank = negative_targets_count.sum()
        median_target_rank = negative_targets_count.median()
        hits_at_1 = (negative_targets_count == 0).sum()
        hits_at_10 = (negative_targets_count < 10).sum()

        logging_output = {
            'target_rank': utils.item(target_rank.data),
            'hits_at_1': utils.item(hits_at_1.data),
            'hits_at_10': utils.item(hits_at_10.data),
            'median_target_rank': utils.item(median_target_rank),  # NOTE: different normalization since it's not a sum
            'normalizer': ntokens
        }

        for l in TrainingMetrics.REPEAT_CONTEXT_LENGTHS:
            total_repeat_at_1, total_wrong_repeat_at_1, total_human_repeat_at_1 = \
                TrainingMetrics.repeat_at_1(logits, targets, context_length=l)

            temp = {'repeat_at_1/%d' % l: utils.item(total_repeat_at_1.data),
                    'wrong_repeat_at_1/%d' % l: utils.item(total_wrong_repeat_at_1.data),
                    'human_repeat_at_1/%d' % l: utils.item(total_human_repeat_at_1.data)
                    }
            for k in temp:
                logging_output[k] = temp[k]

        if topk > 1:
            filtered_topk = top_k_logits(logits, topk)
            softmax_topk = F.softmax(filtered_topk, dim=1)
            true_target_topk_probs = torch.gather(softmax_topk, index=targets[:, None], dim=1).sum()
            logging_output['true_topk_{}_prob'.format(topk)] = true_target_topk_probs.item()
            sum_topk_repeated_probs = 0
            sum_topk_wrepeated_probs = 0
            true_token_zeroed_topk_probs = softmax_topk.clone().scatter_(1, targets[:, None], 0)
            for timestep in range(1, targets.size(0)):
                prev_context = targets[max(0, timestep-128):timestep]
                sum_topk_repeated_probs += torch.gather(softmax_topk[timestep], index=prev_context.unique(), dim=0).sum().item()
                sum_topk_wrepeated_probs += torch.gather(true_token_zeroed_topk_probs[timestep], index=prev_context.unique(), dim=0).sum().item()
            logging_output['repeat_topk_{}'.format(topk)] = sum_topk_repeated_probs
            logging_output['wrepeat_topk_{}'.format(topk)] = sum_topk_wrepeated_probs
            logging_output['nextunique_topk_{}'.format(topk)] = softmax_topk.multinomial(1).view(-1).tolist()

        if topp > 0.0:
            trimmed_topp = SequenceGenerator._sample_topp(SequenceGenerator, F.softmax(logits, dim=1), topp)
            target_mask = (trimmed_topp[1] - targets[:, None].expand(-1, trimmed_topp[1].size(1))) == 0
            true_target_topp_probs = torch.masked_select(trimmed_topp[0], target_mask).sum()
            logging_output['true_topp_{}_prob'.format(topp)] = true_target_topp_probs.item()
            sum_topp_repeated_probs = 0
            sum_topp_wrepeated_probs = 0
            true_token_zeroed_topp_probs = torch.masked_fill(trimmed_topp[0], target_mask, 0)
            for timestep in range(1, targets.size(0)):
                prev_context = targets[max(0, timestep-128):timestep]
                topp_mask = (trimmed_topp[1][timestep][:, None] == prev_context[None, :]).sum(1).nonzero()
                sum_topp_repeated_probs += torch.gather(trimmed_topp[0][timestep], index=topp_mask.view(-1), dim=0).sum().item()
                sum_topp_wrepeated_probs += torch.gather(true_token_zeroed_topp_probs[timestep], index=topp_mask.view(-1), dim=0).sum().item()
            logging_output['repeat_topp_{}'.format(topp)] = sum_topp_repeated_probs
            logging_output['wrepeat_topp_{}'.format(topp)] = sum_topp_wrepeated_probs
            logging_output['nextunique_topp_{}'.format(topp)] = torch.gather(trimmed_topp[1], index=trimmed_topp[0].multinomial(1), dim=1).view(-1).tolist()

        return logging_output

    @staticmethod
    def repeat_at_1(logits, targets, context_length):
        with torch.no_grad():
            predictions = logits.argmax(1)

            targets = targets.unsqueeze(0)
            T = targets.size(1)
            assert logits.size(0) == T

            # T x T where prev_targets[t, :] = [y_1,...,y_t-1, -1, -1,..., -1]
            prev_targets = targets.expand(T, T).tril().masked_fill_(torch.ones_like(targets.expand(T, T)).byte().triu().bool(), -1)

            # each row t is [-1, ..., -1, y_{t-k-1}, ..., y_{t-1}, -1, ..., -1] where k is context length
            prev_targets = prev_targets.masked_fill_(torch.ones_like(targets.expand(T, T)).byte().tril(-(context_length+1)).bool(), -1)

            repeat_at_1 = (predictions[:, None] == prev_targets)
            has_repeat_at_1 = repeat_at_1.sum(1).gt(0)
            total_repeat_at_1 = has_repeat_at_1.sum()

            is_incorrect = (predictions != targets.view(-1)).view(-1, 1)
            total_wrong_repeat_at_1 = ((repeat_at_1 * is_incorrect).sum(1).gt(0)).sum()

            total_human_repeat_at_1 = (prev_targets == targets.view(T, 1)).sum(1).gt(0).sum()

        return total_repeat_at_1, total_wrong_repeat_at_1, total_human_repeat_at_1

    @staticmethod
    def aggregate_and_normalize(logging_outputs):
        agg_output = {}
        normalizer = sum(log.get('normalizer', 0) for log in logging_outputs)
        if normalizer == 0:
            return agg_output
        for name in TrainingMetrics.METRIC_NAMES:
            # 'mean of medians' special case
            if name == 'median_target_rank':
                agg_output[name] = np.mean([log[name] for log in logging_outputs if name in log])
                continue
            metric_sum = sum(log.get(name, 0) for log in logging_outputs)
            metric = metric_sum / normalizer
            agg_output[name] = metric

        # topk and top-p metrics
        keys = set()
        for log in logging_outputs:
            for k in log:
                if 'true_topk' in k or 'true_topp' in k or 'true_full_prob' in k or 'repeat_top' in k:
                    keys.add(k)
        for k in keys:
            metric_sum = sum(log.get(k, 0) for log in logging_outputs)
            metric = metric_sum / normalizer
            agg_output[k] = metric

        unique_top_keys = set()
        for log in logging_outputs:
            for k in log:
                if 'nextunique' in k:
                    unique_top_keys.add(k)
        for k in unique_top_keys:
            unique_list_of_lists = [log.get(k, []) for log in logging_outputs]
            unique_flat_list = []
            for _sublist in unique_list_of_lists:
                unique_flat_list.extend(_sublist)

            unique_metric = len(set(unique_flat_list))
            agg_output[k] = unique_metric

        return agg_output


class Metrics(object):
    def __init__(self, pad=1):
        self._metrics_list = defaultdict(list)
        self._pad = pad

    def reset(self):
        self._metrics_list = defaultdict(list)

    def update(self, batch_of_token_lists):
        if isinstance(batch_of_token_lists, torch.Tensor):
            batch_of_token_lists = batch_of_token_lists.clone().cpu().tolist()

        for token_list in batch_of_token_lists:
            for k, v in ngram_metrics(token_list, pad=self._pad).items():
                self._metrics_list[k].append(v)

    def report(self, kind='train', round_level=4):
        metrics = {}
        # Normalize list-metrics by taking the mean.
        for k, vs in self._metrics_list.items():
            metrics['%s/%s' % (kind, k)] = round(np.mean(vs), round_level)
        return metrics


def ngram_metrics(token_list, pad=1):
    if pad in token_list:
        token_list = token_list[:token_list.index(pad)]  # remove possible padding
    stats = defaultdict(float)
    for n in range(1, 5):
        ngs = [ng for ng in ngrams(token_list, n)]
        counter = Counter([ng for ng in ngrams(token_list, n)])
        stats['pct_repeat_%dgrams' % n] = 1.0 - len(counter)/len(ngs)
    return stats
