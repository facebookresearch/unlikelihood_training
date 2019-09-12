# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from fairseq import options, sequence_generator
from fairseq.custom import evaluate_utils
import argparse
from glob import glob
import os.path
import getpass
import sys
import shlex
import pickle
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    script_parser = argparse.ArgumentParser(description='Computes greedy completion, single-token prediction, and corresponding targets.')
    script_parser.add_argument('--data-dir', type=str, required=True)
    script_parser.add_argument('--base-dir', type=str, required=True)
    script_parser.add_argument('--eval-mode', choices=['all', 'completion', 'singletoken'], default='all')
    script_parser.add_argument('--data-prefix-length', type=int, default=50, help='Length of prefix')
    script_parser.add_argument('--batch-size-completions', type=int, default=128)
    script_parser.add_argument('--batch-size-single-prediction', type=int, default=1024)

    script_parser.add_argument('--completion-length', type=int, default=500,
                               help='The length of each generated sequence, not counting the prefix length')
    script_parser.add_argument('--model-path', type=str, required=True, help='The path to the folder with checkpoints')
    script_parser.add_argument('--save-path', type=str, required=True)
    script_parser.add_argument('--ckpt', choices=['best', 'last', 'all', 'step', 'epoch'], default='best')
    script_parser.add_argument('--ckpt-step', type=str, default=None)
    script_parser.add_argument('--ckpt-epoch', type=str, default=None)
    script_parser.add_argument('--data-split', choices=['train', 'valid', 'test'], default='valid')
    script_parser.add_argument('--num-samples', type=int, default=-1)
    script_parser.add_argument('--beam-size', type=int, default=1)
    script_parser.add_argument('--beam-ngram-block', type=int, default=0)
    script_parser.add_argument('--topp', type=float, default=0.0)
    script_parser.add_argument('--topk', type=int, default=1)
    script_parser.add_argument('--singletoken-topk', type=int, default=1)
    script_parser.add_argument('--singletoken-topp', type=float, default=0.0)

    high_level_args = script_parser.parse_args()

    if high_level_args.ckpt == 'last':
        checkpoints = glob(os.path.join(high_level_args.model_path, 'checkpoint_last.pt'))
    elif high_level_args.ckpt == 'best':
        checkpoints = glob(os.path.join(high_level_args.model_path, 'checkpoint_best.pt'))
    elif high_level_args.ckpt == 'step':
        checkpoints = glob(os.path.join(high_level_args.model_path, 'checkpoint_*_{}.pt'.format(high_level_args.ckpt_step)))
    elif high_level_args.ckpt == 'epoch':
        checkpoints = glob(
            os.path.join(high_level_args.model_path, 'checkpoint{}.pt'.format(high_level_args.ckpt_epoch)))
    elif high_level_args.ckpt == 'all':
        checkpoints = glob(os.path.join(high_level_args.model_path, 'checkpoint*'))

    print("Evaluating {} checkpoints.".format(len(checkpoints)))
    for i, checkpoint in enumerate(checkpoints):

        if high_level_args.eval_mode in ['all', 'completion']:
            num_tokens = high_level_args.data_prefix_length*high_level_args.batch_size_completions
            FAIRSEQ_OPTS = "--data {} \
                            --task language_modeling_with_generation \
                            --path {} \
                            --tokens-per-sample {} \
                            --max-tokens {} \
                            --sample-break-mode none \
                            --gen-subset {} \
                            --user-dir {}".format(high_level_args.data_dir, checkpoint,
                                                  num_tokens, num_tokens, high_level_args.data_split,
                                                  os.path.join(high_level_args.base_dir, 'fairseq/custom'))
            sys.argv = shlex.split(FAIRSEQ_OPTS)
            parser = options.get_generation_parser()
            args = options.parse_args_and_arch(parser)
            args.add_bos_token = False
            args.skip_invalid_size_inputs_valid_test = False

            task, model, generator, itr, step = evaluate_utils.load(args)

            task.dictionary.eos_index = len(task.dictionary) - 1
            task.dictionary.eos_word = task.dictionary.symbols[-1]

            fairseq_generator = sequence_generator.SequenceGenerator(tgt_dict=task.dictionary,
                                                                     beam_size=high_level_args.beam_size,
                                                                     no_repeat_ngram_size=high_level_args.beam_ngram_block,
                                                                     max_len_b=high_level_args.completion_length+high_level_args.data_prefix_length,
                                                                     )

            filename_suffix = '_{}__st_{}__spl_{}__pfx_{}__cmpl_{}__bs_cmpl_{}__bs_sprd_{}__bms_{}__bnb_{}__tpk_{}__tpp_{}__sttpk_{}__sttpp_{}__ckst_{}__ckep_{}__ckpt_{}'.format(
                os.path.basename(os.path.normpath(high_level_args.model_path)),
                step, high_level_args.data_split, high_level_args.data_prefix_length, high_level_args.completion_length,
                high_level_args.batch_size_completions, high_level_args.batch_size_single_prediction,
                high_level_args.beam_size, high_level_args.beam_ngram_block, high_level_args.topk, high_level_args.topp, high_level_args.singletoken_topk,
                high_level_args.singletoken_topp, high_level_args.ckpt_step, high_level_args.ckpt_epoch,
                high_level_args.ckpt)

            completions, gen_metrics, actual_metrics = evaluate_utils.generate_completions(
                model, generator, fairseq_generator, itr,
                high_level_args.data_prefix_length,
                high_level_args.completion_length,
                topk=high_level_args.topk,
                beam_size=high_level_args.beam_size,
                num_samples=high_level_args.num_samples,
                topp=high_level_args.topp)

            completion_tokens = [[task.dictionary[i] for i in sample] for sample in completions]
            completion_text = [' '.join(ts) for ts in completion_tokens]

            # dump generation to text file
            completion_output_filename = os.path.join(high_level_args.save_path,
                                                      'completions_{}.txt'.format(filename_suffix))
            with open(completion_output_filename, 'w') as f:
                for line in completion_text:
                    f.write(line)
                    f.write('\n')
                print("\tcompletions output file: %s" % completion_output_filename)


        if high_level_args.eval_mode in ['all', 'singletoken']:
            num_tokens = high_level_args.batch_size_single_prediction
            FAIRSEQ_OPTS = "--data {} \
                                        --task language_modeling_with_generation \
                                        --path {} \
                                        --tokens-per-sample {} \
                                        --max-tokens {} \
                                        --sample-break-mode none \
                                        --gen-subset {} \
                                        --user-dir {}".format(high_level_args.data_dir, checkpoint,
                                                              num_tokens, num_tokens, high_level_args.data_split,
                                                              os.path.join(high_level_args.base_dir,
                                                                           'fairseq/custom'))
            sys.argv = shlex.split(FAIRSEQ_OPTS)
            parser = options.get_generation_parser()
            args = options.parse_args_and_arch(parser)
            args.add_bos_token = False
            args.skip_invalid_size_inputs_valid_test = False

            task, model, generator, itr, step = evaluate_utils.load(args)

            single_predicted_tokens, target_tokens, metrics = evaluate_utils.eval_single_token_prediction(
                model, itr, task.target_dictionary, singletoken_topk=high_level_args.singletoken_topk,
                singletoken_topp=high_level_args.singletoken_topp)

            subset_metrics = {}
            subset_data = high_level_args.data_split

            for metric_name, value in metrics.items():
                subset_metrics[f'{subset_data}/{metric_name}'] = value
            subset_metrics['checkpoint_step'] = step

            filename_suffix = '_{}__st_{}__spl_{}__pfx_{}__cmpl_{}__bs_cmpl_{}__bs_sprd_{}__bms_{}__bnb_{}__tpk_{}__tpp_{}__sttpk_{}__sttpp_{}__ckst_{}__ckep_{}__ckpt_{}'.format(
                os.path.basename(os.path.normpath(high_level_args.model_path)),
                step, high_level_args.data_split, high_level_args.data_prefix_length, high_level_args.completion_length,
                high_level_args.batch_size_completions, high_level_args.batch_size_single_prediction,
                high_level_args.beam_size, high_level_args.beam_ngram_block, high_level_args.topk, high_level_args.topp,
                high_level_args.singletoken_topk,
                high_level_args.singletoken_topp, high_level_args.ckpt_step, high_level_args.ckpt_epoch,
                high_level_args.ckpt)

            single_token_predictions_filename = os.path.join(high_level_args.save_path, "single_token_predictions_{}.txt".format(filename_suffix))

            pkl_filename = os.path.join(high_level_args.save_path, "metrics_{}.pkl".format(filename_suffix))
            pickle.dump(subset_metrics, open(pkl_filename, 'wb'))

            with open(single_token_predictions_filename, 'w') as f:
                for single_predicted_tokens_sublist in single_predicted_tokens:
                    _single_token_text = [task.dictionary[i] for i in single_predicted_tokens_sublist]
                    f.write(' '.join(_single_token_text))
                    f.write('\n')

            target_filename = os.path.join(high_level_args.save_path, "targets_{}.txt".format(filename_suffix))

            with open(target_filename, 'w') as f:
                for target_tokens_sublist in target_tokens:
                    _target_text = [task.dictionary[i] for i in target_tokens_sublist]
                    f.write(' '.join(_target_text))
                    f.write('\n')


if __name__ == '__main__':
    main()
