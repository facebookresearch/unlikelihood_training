# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter
from fairseq.custom.metrics import Metrics, TrainingMetrics
from tqdm import tqdm
import torch.nn.functional as F
import math

def load(args, task=None, itr=None, generator=None, log=False):
    """Returns task, model, generator, and dataset iterator for the given `args`."""
    assert args.path is not None, '--path required for generation!'
    import random
    random.seed(42)
    torch.manual_seed(42)
    utils.import_user_module(args)
    if log:
        print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    if task is None:
        task = tasks.setup_task(args)
        task.load_dataset(args.gen_subset)

    # Load ensemble
    if log:
        print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
    model = models[0]

    if itr is None:
        # Load dataset (possibly sharded)
        itr = task.get_batch_iterator(
            dataset=task.dataset(args.gen_subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=args.tokens_per_sample,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

    # Get model step
    step = torch.load(args.path)['optimizer_history'][-1]['num_updates']

    if generator is None:
        # Initialize generator
        generator = task.build_generator(args)
    return task, model, generator, itr, step


def generate_completions(model, generator, fairseq_generator, itr, eval_prefix_length, eval_completion_length, topk, topp, num_samples, beam_size, include_prefix=True):
    completions = []
    completion_metrics = Metrics()
    actual_metrics = Metrics()
    for n, sample in enumerate(tqdm(itr)):
        input_sequence = sample['net_input']['src_tokens']
        prefix_batch = batch_input_sequence_by_prefix_length(input_sequence, eval_prefix_length)
        prefix_batch = prefix_batch.cuda()
        if input_sequence.size(1) < eval_prefix_length:
            continue
        if beam_size > 1:
            assert topk == 1, 'with greedy topk must be 1'
            assert topp == 0.0, 'with greedy topp must be 0'
            sample['net_input']['src_tokens'] = prefix_batch
            res = fairseq_generator.generate([model], sample, prefix_batch, bos_token=0)  # prefix is there in preds!
            pred_completion = [res[i][0]['tokens'][eval_prefix_length:-1].cpu().tolist() for i in range(len(res))]
        elif beam_size == 1:
            pred_completion = generator.generate_completion(model, prefix_batch, eval_completion_length, topk, topp)
            pred_completion = pred_completion.cpu().tolist()
        completion_metrics.update(pred_completion)
        actual_metrics.update(input_sequence)

        if include_prefix:
            prefix_batch = prefix_batch.cpu().tolist()
            pred_completion = [prefix + completion for
                               prefix, completion in zip(prefix_batch, pred_completion)]
        completions.extend(pred_completion)

        if n == num_samples:
            break

    completion_metrics = completion_metrics.report('generated')
    actual_metrics = actual_metrics.report('actual')
    return completions, completion_metrics, actual_metrics


def batch_input_sequence_by_prefix_length(input_sequence, prefix_length):
    seq_len = input_sequence.size(1)
    # Discard tokens if the sequence length is not divisible by the prefix length.
    new_seq_len = (seq_len//prefix_length)*prefix_length
    input_sequence = input_sequence[:, :new_seq_len]
    batch = input_sequence.view(-1, prefix_length).contiguous()
    return batch


@torch.no_grad()
def eval_single_token_prediction(model, itr, dictionary, singletoken_topp=0.0, singletoken_topk=1):
    predicted_tokens = []
    target_tokens = []

    mle_loss_sum = 0
    num_samples_sum = 0
    wrong_mass_sum = 0

    logging_outputs = []

    for n, sample in tqdm(enumerate(itr)):
        sample = utils.move_to_cuda(sample)
        net_output = model(**sample['net_input'])
        logits = net_output[0][0]
        logits[:, dictionary.pad()] = -1e19
        predicted_tokens.append(logits.argmax(1).tolist())
        target = sample['target'].view(-1)
        target_tokens.append(target.tolist())

        # -- mle loss
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        true_token_lprobs = F.nll_loss(
            lprobs,
            target,
            ignore_index=dictionary.pad_index,
            reduction='none',
        )
        true_token_logits = -F.nll_loss(
            logits,
            target,
            ignore_index=dictionary.pad_index,
            reduction='none',
        )
        mle_loss = true_token_lprobs.sum()
        orig = utils.strip_pad(target, dictionary.pad_index)
        ntokens = orig.numel()

        mle_loss_sum += mle_loss.item()
        num_samples_sum += ntokens

        logging_output = TrainingMetrics.ranking_metrics(logits, true_token_logits, sample, ntokens, target, topk=singletoken_topk, topp=singletoken_topp)

        negative_targets = (logits > true_token_logits[:, None]).float()
        wrong_mass_sum += (negative_targets * (F.softmax(logits, dim=1))).sum()

        logging_outputs.append(logging_output)

    ppl = math.pow(2, mle_loss_sum / num_samples_sum / math.log(2))
    custom_metrics = TrainingMetrics.aggregate_and_normalize(logging_outputs)
    custom_metrics['ppl'] = ppl
    avg_wrong_mass = wrong_mass_sum / num_samples_sum
    custom_metrics['avg_wrong_mass'] = avg_wrong_mass.item()
    return predicted_tokens, target_tokens, custom_metrics