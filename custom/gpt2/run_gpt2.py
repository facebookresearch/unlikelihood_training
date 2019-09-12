# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import logging
import json
import os
import re
import random

import torch
import torch.nn.functional as F
import numpy as np

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, WarmupLinearSchedule, WEIGHTS_NAME, CONFIG_NAME
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler

from fairseq.custom.metrics import TrainingMetrics, Metrics, ngram_metrics
from fairseq.custom.baseline_cross_entropy import CrossEntropyCriterionWCustomMetrics
from fairseq.custom.sequence_penalty_loss import SequencePenaltyCriterion
from fairseq.custom.evaluate_utils import batch_input_sequence_by_prefix_length

from collections import defaultdict
from tqdm import tqdm, trange
from pprint import pprint

RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.size(0) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    logits = logits.squeeze(0)
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def get_datasets(dataset_paths, max_len=1536):
    """Args:
        dataset_paths: {'train': str, 'valid': str, 'test': str}
    """
    datasets = {}

    for split, fname in dataset_paths.items():
        tensor = torch.load(fname)
        right_bound = (tensor.size(0) // (max_len+1)) * (max_len + 1)
        dataset = TensorDataset(tensor[:right_bound].view(-1, (max_len+1)))
        datasets[split] = dataset

    return datasets


def sample_sequence(model, prefix_batch, prefix_length, continuation_length, top_k, top_p):
    continuation_logits = []
    context = prefix_batch
    assert context.size(1) == prefix_length

    prev = context
    output = context
    past = None
    for i in range(continuation_length):
        logits, past = model(prev, past=past)
        logits = logits[:, -1, :]
        if top_k == 1 and top_p == 0:
            prev = logits.argmax(dim=1, keepdim=True)
        else:
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            prev = F.softmax(filtered_logits, dim=-1).multinomial(num_samples=1)

        continuation_logits.append(logits)
        output = torch.cat((output, prev), dim=1)

    continuation_logits = torch.stack(continuation_logits, 1)
    return output, continuation_logits


def mle_loss(model, batch, args):
    longer_sample = batch[0].cuda()
    inp = longer_sample[:, :args.train_batch_size]
    model_output = model(inp)
    target = longer_sample[:, 1:]
    logits = model_output[0]
    lprobs = F.log_softmax(logits, dim=-1)
    assert lprobs.size(0) == 1, 'We work on flat sequences'
    loss = F.nll_loss(lprobs[0], target[0], reduction='sum')
    true_token_logits = -F.nll_loss(logits[0], target[0], reduction='none')
    ntokens = inp.numel()

    logging_output = TrainingMetrics.ranking_metrics(logits[0], true_token_logits, None, ntokens, target[0])
    logging_output['loss'] = loss.item()
    logging_output['normalizer'] = ntokens
    logging_output['sample_size'] = ntokens
    logging_output['ntokens'] = ntokens

    loss = loss / ntokens
    return loss, logging_output


def ul_seq(model, batch, args):
    input_sequence = batch[0].cuda()
    batch = batch_input_sequence_by_prefix_length(input_sequence, args.prefix_length)
    completions, continuation_logits = sample_sequence(model, batch,
                                                       args.prefix_length, args.continuation_length, args.top_k, args.top_p)
    pred_toks = completions[:, args.prefix_length:].contiguous()

    mask = ngram_repeat_mask(pred_toks, args.sequence_ngram_n).type_as(continuation_logits)

    lprobs = F.log_softmax(continuation_logits, dim=-1)
    pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
    one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
    loss = -torch.log(one_minus_probs) * mask
    loss = loss.sum()
    ntokens = pred_toks.numel()  # number of output tokens (tokens in completions)

    logging_output = {
        'seq_loss': loss.item(),
        'seq_sample_size': ntokens,
        'seq_ntokens': ntokens,
        'seq_nsentences': batch.size(0),
        'seq_repeat_mask': mask.sum().item(),
    }

    # Sum each statistic, which will be normalized by the number of sentences in `aggregate_logging_outputs`.
    stats = defaultdict(float)
    for tok_list in pred_toks.cpu().tolist():
        ms = ngram_metrics(tok_list)
        for k, v in ms.items():
            stats[k] += v
    for k, v in stats.items():
        logging_output[k] = v

    loss = loss / ntokens
    return loss, logging_output


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


def tokenize(text):
    # ref: https://github.com/facebookresearch/ParlAI/blob/4da3ec0bdcf1db2c3a5bd5723d1275c32a891192/parlai/core/dict.py#L451
    return RETOK.findall(text)


def get_text_continuation(bpe_completion, tokenizer, args):
    completion = tokenizer.decode(bpe_completion)
    bpe_prefix, bpe_continuation = bpe_completion[:args.prefix_length], bpe_completion[args.prefix_length:]
    prefix = tokenizer.decode(bpe_prefix)

    if prefix in completion:
        continuation = completion.replace(prefix, '')
    else:
        prefix_ = ' '.join(prefix.split(' ')[:-2])
        continuation = completion.replace(prefix_, '')

    continuation_tokens = tokenize(continuation)
    return continuation_tokens


def save_completion_metrics(bpe_metrics, word_metrics, text_completions, config, args):
    outfile = os.path.join(args.output_dir,
                           'completion__{model}__spl_{split}__topk_{topk}__topp_{topp}__pfl_{pfl}__cnl_{cnl}'.format(
                               model=args.model_name,
                               split=args.eval_split,
                               topk=args.top_k,
                               topp=args.top_p,
                               pfl=args.prefix_length,
                               cnl=args.continuation_length
                           ))
    json.dump({'bpe_metrics': bpe_metrics,
               'word_metrics': word_metrics,
               'config': config,
               'completions': text_completions}, open(outfile + '.json', 'w'))
    print("%s metrics written to %s" % (args.mode, outfile + '.json'))


def save_singletoken_metrics(metrics, config, args, best=False, train_iter=None):
    output_dir = args.output_dir if not best else os.path.join(args.output_dir, 'best')
    outfile = os.path.join(output_dir,
                           'singletoken__{model}__spl_{split}__bsz_{bsz}{iter}.json'.format(
                               model=args.model_name,
                               split=args.eval_split,
                               bsz=args.batch_size_singletoken,
                               iter='_%d' % train_iter if train_iter is not None else '',
                           ))

    json.dump({'metrics': metrics,
               'config': config}, open(outfile, 'w'))
    print("%s metrics written to %s" % (args.mode, outfile))


def eval_singletoken(model, args, dataset_paths, train_iter=None):
    datasets = get_datasets(dataset_paths, max_len=args.batch_size_singletoken)
    eval_sampler = SequentialSampler(datasets[args.eval_split])
    eval_dataloader = DataLoader(datasets[args.eval_split], sampler=eval_sampler, batch_size=1)

    model.eval()

    logging_outputs = []
    predicted_tokens = []
    target_tokens = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating", total=len(eval_dataloader)):
            longer_sample = batch[0].cuda()
            inp = longer_sample[:, :args.batch_size_singletoken]
            model_output = model(inp)
            target = longer_sample[:, 1:]
            logits = model_output[0]
            lprobs = F.log_softmax(logits, dim=-1)
            assert lprobs.size(0) == 1, 'We work on flat sequences'
            loss = F.nll_loss(lprobs[0], target[0], reduction='sum')
            true_token_logits = -F.nll_loss(logits[0], target[0], reduction='none')

            pred = lprobs.argmax(dim=-1).view(-1).tolist()
            predicted_tokens.extend(pred)
            ntokens = inp.numel()

            logging_output = TrainingMetrics.ranking_metrics(logits[0], true_token_logits, None, ntokens, target[0])
            logging_output['loss'] = loss.item()
            logging_output['normalizer'] = ntokens
            logging_output['sample_size'] = ntokens
            logging_output['ntokens'] = ntokens
            logging_outputs.append(logging_output)

            # for human uniq
            target_tokens.extend(target.view(-1).tolist())

    logging_average = CrossEntropyCriterionWCustomMetrics.aggregate_logging_outputs(logging_outputs)
    logging_average['ppl'] = 2 ** logging_average['loss']
    logging_average['uniq'] = len(set(predicted_tokens))
    logging_average['human_uniq'] = len(set(target_tokens))

    save_singletoken_metrics(logging_average, model.config.to_dict(), args, train_iter=train_iter)
    return logging_average


def main():
    parser = argparse.ArgumentParser(description='openGPT-2 analysis')

    parser.add_argument('--mode', choices=['train', 'eval-singletoken', 'eval-completion', 'eval-both'], default='eval-singletoken')
    parser.add_argument('--eval-split', choices=['train', 'valid', 'test'])
    parser.add_argument('--model-name', choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2-medium')
    parser.add_argument('--model-load-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-base', type=str)
    parser.add_argument('--num-train-epochs', type=int, default=1)
    parser.add_argument('--batch-size-singletoken', type=int, default=1024)
    parser.add_argument('--batch-size-completion', type=int, default=300)
    parser.add_argument("--output-dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # eval-completion
    parser.add_argument('--prefix-length', type=int, default=50)
    parser.add_argument('--continuation-length', type=int, default=100)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--top-p', type=float, default=0.0)

    # custom training
    parser.add_argument('--sequence-tune-rate', type=float, default=0.5)
    parser.add_argument('--train-batch-size', type=int, default=300)
    parser.add_argument('--report-metrics-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--sequence-ngram-n', type=int, default=4)
    parser.add_argument('--train-n-steps', type=int, default=10000)
    parser.add_argument('--validate-every', type=int, default=10000)

    # training loop
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max-grad-norm', type=int, default=1)
    parser.add_argument("--max-steps", default=-1, type=int,
                        help="If > 0: set total number of training \
                            steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help="Number of updates steps to accumulate before\
                            performing a backward/update pass.")
    parser.add_argument('--learning-rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup-steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--lr-schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--lm-coef', type=float, default=0.9)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    dataset_paths = {
        'train': os.path.join(args.data_base, 'train_tokens_bpe_gpt2.pt'),
        'valid': os.path.join(args.data_base, 'valid_tokens_bpe_gpt2.pt'),
        'test': os.path.join(args.data_base, 'test_tokens_bpe_gpt2.pt'),
    }

    if args.model_load_dir:
        model = GPT2LMHeadModel.from_pretrained(args.model_load_dir)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(device)

    if args.mode == 'eval-singletoken' or args.mode == 'eval-both':
        eval_singletoken(model, args, dataset_paths)

    if args.mode == 'eval-completion' or args.mode == 'eval-both':
        datasets = get_datasets(dataset_paths, max_len=args.batch_size_completion)
        eval_sampler = SequentialSampler(datasets[args.eval_split])
        eval_dataloader = DataLoader(datasets[args.eval_split], sampler=eval_sampler, batch_size=1)

        model.eval()

        with torch.no_grad():
            all_text_completions = []

            bpe_ngram_metrics = Metrics(pad=-1)
            word_ngram_metrics = Metrics(pad=-1)

            for i, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating", total=len(eval_dataloader)):
                input_sequence = batch[0].cuda()
                if input_sequence.size(1) < args.prefix_length:
                    continue

                # Predict the completions.
                batch = batch_input_sequence_by_prefix_length(input_sequence, args.prefix_length)
                bpe_completions, _ = sample_sequence(model, batch, args.prefix_length, args.continuation_length, args.top_k, args.top_p)
                bpe_completions = bpe_completions.tolist()

                # Extract continuations from the predicted completions.
                bpe_continuations = []
                text_continuations = []
                for bpe_completion in bpe_completions:
                    bpe_continuations.append(bpe_completion[args.prefix_length:])
                    text_continuations.append(get_text_continuation(bpe_completion, tokenizer, args))
                    all_text_completions.append(tokenizer.decode(bpe_completion))

                # Only keep continuations with at least one 4-gram
                # (A short continuation may occur due to predicted whitespace, then tokenizing, despite being
                #  normal length in BPE tokens).
                text_continuations = [c for c in text_continuations if len(c) > 3]

                # Update metrics with this batch of continuations.
                bpe_ngram_metrics.update(bpe_continuations)
                word_ngram_metrics.update(text_continuations)

                # Save the (possibly intermediate) metrics.
                save_completion_metrics(bpe_metrics=bpe_ngram_metrics.report('bpe_%s' % args.eval_split),
                                        word_metrics=word_ngram_metrics.report('word_%s' % args.eval_split),
                                        text_completions=all_text_completions,
                                        config=model.config.to_dict(),
                                        args=args)

    if args.mode == 'train':
        if not os.path.exists(os.path.join(args.output_dir, 'best')):
            os.makedirs(os.path.join(args.output_dir, 'best'))

        token_loss = mle_loss
        datasets = get_datasets(dataset_paths, max_len=args.train_batch_size)
        train_sampler = RandomSampler(datasets['train'])
        train_seq_dataloader = DataLoader(datasets['train'], sampler=train_sampler, batch_size=1)

        # Setup optimizer
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_seq_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_seq_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

        total_steps = 0
        best_ppl = 1e20
        for _ in trange(args.num_train_epochs, desc="Epoch"):
            logging_outputs = []
            epoch_loss = 0
            epoch_steps = 0
            tqdm_bar = tqdm(train_seq_dataloader, desc="Training", total=args.train_n_steps)
            for step, batch in enumerate(tqdm_bar):
                optimizer.zero_grad()

                # Sequence loss
                if torch.rand(1).item() < args.sequence_tune_rate:
                    if batch[0].size(1) < args.prefix_length:
                        continue
                    loss, batch_metrics = ul_seq(model, batch, args)

                # Token loss
                else:
                    loss, batch_metrics = token_loss(model, batch, args)

                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(epoch_loss/epoch_steps, scheduler.get_lr()[0])

                logging_outputs.append(batch_metrics)

                if epoch_steps % args.report_metrics_every == 0:
                    logging_average = CrossEntropyCriterionWCustomMetrics.aggregate_logging_outputs(logging_outputs)
                    temp = SequencePenaltyCriterion.aggregate_logging_outputs(logging_outputs)
                    for k, v in temp.items():
                        logging_average[k] = v
                    logging_average['ppl'] = 2 ** logging_average['loss']
                    print(logging_average)
                    logging_outputs = []

                if step == args.train_n_steps:
                    break

                if epoch_steps % args.save_every == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(args.output_dir)

                if total_steps % args.validate_every == 0:
                    print("Validating...")
                    validation_outputs = eval_singletoken(model, args, dataset_paths, train_iter=total_steps)
                    if validation_outputs['ppl'] < best_ppl:
                        best_ppl = validation_outputs['ppl']
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(args.output_dir, 'best', WEIGHTS_NAME)
                        output_config_file = os.path.join(args.output_dir, 'best', CONFIG_NAME)
                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(os.path.join(args.output_dir, 'best'))
                        save_singletoken_metrics(validation_outputs, model.config.to_dict(), args,
                                                 train_iter=total_steps, best=True)


if __name__ == '__main__':
    main()
