# Neural Text ~~de~~Generation with Unlikelihood Training

PyTorch implementation of the paper:

[Neural Text Generation with Unlikelihood Training](https://arxiv.org/pdf/1908.04319.pdf)\
Sean Welleck\*, Ilia Kulikov\*, Stephen Roller, Emily Dinan, Kyunghyun Cho, Jason Weston\
\*Equal contribution. The order was decided by a coin flip.

We present code for training models described in the paper, as well as pre-trained models. The code includes:
- An **implementation of unlikelihood training, fine-tuning, and evaluation** for [fairseq](https://github.com/pytorch/fairseq).
- A script for **fine-tuning a GPT-2 model** from [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) with the unlikelihood sequence loss.

| Table of Contents |
|-|
| [Setup](#setup)|
| [Training](#training)|
| [Evaluation](#evaluation)|
| [Finetuning GPT-2](#finetuning-gpt-2)|

Please cite our work if you found the resources in this repository useful:
```
@misc{welleck2019neural,
    title={Neural Text Generation with Unlikelihood Training},
    author={Sean Welleck and Ilia Kulikov and Stephen Roller and Emily Dinan and Kyunghyun Cho and Jason Weston},
    year={2019},
    eprint={1908.04319},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Setup
### Dependencies
The implementation is a custom [fairseq](https://github.com/pytorch/fairseq) module. Download and install fairseq:
```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout 2b68e91f231a2b7997664e1418f30b808d889963
pip install --editable .
```

Install other dependencies:
```bash
pip install nltk
pip install pandas
pip install pytorch-transformers   # (optional); for GPT-2 fine-tuning
pip install tensorflow=1.14
pip install tensorboardX           # (optional); for tensorboard logs
pip install torch==1.4.0           # overwriting the latest version of pytorch, as installed by fairseq
```

### 'Installing' the unlikelihood module
Copy the `custom` directory in this repo into the `fairseq` repo that you downloaded above:
```bash
export FAIRSEQ_DIR=/path/to/fairseq
export UNLIKELIHOOD_DIR=/path/to/unlikelihood_training

cp -r $UNLIKELIHOOD_DIR/custom $FAIRSEQ_DIR/fairseq
```
Now `ls $FAIRSEQ_DIR/fairseq` should resemble:
```bash
binarizer.py
...
criterions
custom
data
...
```

### Next Steps
We recommend performing the following steps **from the `fairseq` repo's base directory**:
```bash
cd $FAIRSEQ_DIR
```

### Dataset

Download the binarized wikitext-103 dataset (160MB, install `wget` if needed):
```bash
wget https://dl.fbaipublicfiles.com/unlikelihood/wikitext-103_v0.tar.gz
```

Unpack the dataset (440MB):
```bash
tar xzvf wikitext-103_v0.tar.gz
```
This command unpacks the dataset into a `data-bin` folder in the current directory.

### Create a checkpoint folder
```
mkdir checkpoint
```

### Download pre-trained models
\*This step is not necessary for *training* a model from scratch.

We provide all `fairseq` models used in the paper. Download the model archive (*warning*: large (16gb) file):
```bash
wget https://dl.fbaipublicfiles.com/unlikelihood/checkpoints_v0.tar.gz
```
Unpack the model checkpoints from the archive:
```bash
tar xzvf checkpoints_v0.tar.gz
```

## Training
\*We tested these scripts using Tesla V100 32GB gpu(s) in both single and multi-gpu (8) settings. If you get OOM errors, try decreasing the batch size (`--max-tokens`,`--tokens-per-sample`). Otherwise, the hyper-parameters used here are similar to the example LM training code in `fairseq`.

The commands below assume you are in the `$FAIRSEQ_DIR` directory.
### Baseline (MLE) model

```bash
python -u ./train.py --task language_modeling_with_generation ./data-bin/wikitext-103 \
    --user-dir ./fairseq/custom --arch transformer_lm_ul --max-tokens 1536 --tokens-per-sample 1536 \
    --fp16 --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 \
    --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 \
    --optimizer nag --lr 0.0001 --clip-norm 0.1 --update-freq 3 --seed 1 --sample-break-mode none \
    --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --save-interval-updates 10000 \
    --keep-interval-updates 2 --no-progress-bar --log-interval 100 \
    --criterion cross_entropy_wcustom_metrics \
    --save-dir ./checkpoint/baseline_model \
    --tensorboard-logdir ./checkpoint/baseline_model
```
### Train a token-level unlikelihood model

```bash
python -u ./train.py --task language_modeling_with_generation ./data-bin/wikitext-103 \
    --user-dir ./fairseq/custom --arch transformer_lm_ul --max-tokens 1536 --tokens-per-sample 1536 \
    --fp16 --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 \
    --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 \
    --optimizer nag --lr 0.0001 --clip-norm 0.1 --update-freq 3 --seed 1 --sample-break-mode none \
    --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --save-interval-updates 10000 \
    --keep-interval-updates 2 --no-progress-bar --log-interval 100 \
    --criterion candidate_penalty_cross_entropy --rank-alpha 1.0 \
    --save-dir ./checkpoint/token_level_model \
    --tensorboard-logdir ./checkpoint/token_level_model
```

### Sequence-level fine tuning

For sequence-level fine tuning you need an initial checkpoint (via `--restore-file`). You can use your own checkpoints, or a provided checkpoint as shown below.

#### Fine-tuning the baseline model

```bash
python -u ./train.py --task language_modeling_with_generation ./data-bin/wikitext-103 \
    --user-dir ./fairseq/custom --arch transformer_lm_ul --max-tokens 1536 --tokens-per-sample 1536 \
    --fp16 --max-update 1500 --max-lr 1.0e-2 --t-mult 2 --lr-period-updates 270000 \
    --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 0 --warmup-init-lr 1e-07 --min-lr 1e-09 \
    --optimizer nag --lr 0.0001 --clip-norm 0.1 --update-freq 3 --seed 1 --sample-break-mode none \
    --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --save-interval-updates 100 \
    --keep-interval-updates 2 --no-progress-bar --log-interval 10 \
    --rank-alpha 1.0 --sequence-level-train-rate 0.5 \
    --reset-lr-scheduler --reset-optimizer --reset-meters \
    --compute-metrics-interval 1 --restore-file ./public_checkpoints/mle_baseline/checkpoint_best.pt \
    --criterion cross_entropy_wcustom_metrics \
    --sequence-prefix-length 50 --sequence-completion-length 100 \
    --sequence-ngram-n 4 \
    --save-dir ./checkpoint/seq_level_on_baseline \
    --tensorboard-logdir ./checkpoint/seq_level_on_baseline
```

#### Fine-tuning the token-level unlikelihood model
```bash
python -u ./train.py --task language_modeling_with_generation ./data-bin/wikitext-103 \
    --user-dir ./fairseq/custom --arch transformer_lm_ul --max-tokens 1536 --tokens-per-sample 1536 \
    --fp16 --max-update 1500 --max-lr 1.0e-2 --t-mult 2 --lr-period-updates 270000 \
    --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 0 --warmup-init-lr 1e-07 --min-lr 1e-09 \
    --optimizer nag --lr 0.0001 --clip-norm 0.1 --update-freq 3 --seed 1 --sample-break-mode none \
    --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --save-interval-updates 100 \
    --keep-interval-updates 2 --no-progress-bar --log-interval 10 \
    --rank-alpha 1.0 --sequence-level-train-rate 0.5 \
    --reset-lr-scheduler --reset-optimizer --reset-meters \
    --compute-metrics-interval 1 --restore-file ./public_checkpoints/token_level_ul/checkpoint_best.pt \
    --criterion candidate_penalty_cross_entropy \
    --sequence-prefix-length 50 --sequence-completion-length 100 \
    --sequence-ngram-n 4 \
    --save-dir ./checkpoint/seq_level_on_token_level \
    --tensorboard-logdir ./checkpoint/seq_level_on_token_level
```

## Evaluation

A single script (`custom/evaluation.py`) performs sequence-level and token level evaluation. For the sequence-level evaluation one can choose greedy search, beam search, top-k, or top-p (nucleus) sampling.

Each evaluation run produces the following files (in the `--save-path` directory):

- `completions__{params}.txt`: prefixes with corresponding completions
- `single_token_predictions__{params}.txt`: next-token greedy predictions (i.e. given human context) 
- `metrics__{params}.pkl`: metrics extracted on the token-level (e.g. PPL, loss, acc, rep, etc.)
- `targets__{params}.txt`: reference sequences

Example command to run evaluation using the pretrained baseline model:

```bash
python -u ./fairseq/custom/evaluation.py \
    --batch-size-single-prediction 1536 --batch-size-completion 48 \
    --data-prefix-length 50 --completion-length 100 \
    --save-path ./public_checkpoints/ --ckpt all \
    --model-path ./public_checkpoints/mle_baseline \
    --data-dir ./data-bin/wikitext-103 \
    --base-dir ./
```

#### Evaluation from the paper
We share evaluation outputs for models used in our paper. To download and unpack the outputs:
```
wget https://dl.fbaipublicfiles.com/unlikelihood/eval_public_v0.tar.gz
tar xzvf eval_public_v0.tar.gz
```

To post-process evaluation output (requires `pandas` (`pip install pandas`)):
```bash
python fairseq/custom/report_metrics.py \
    --eval-dir ./eval_public \
    --model-names mle_baseline token_level_ul seq_level_ul_mle seq_level_ul_token_level_ul
```
This yields the following output:
```
     model_name beam size beam block topk topp  split  seq-rep-1  seq-rep-4  uniq-seq     ppl    acc    rep   wrep   uniq
0  mle_baseline         1          0   50  0.0  valid      0.381      0.016     21396  24.592  0.401  0.619  0.346  11654
1  mle_baseline         1          0    1  0.0  valid      0.690      0.429     10629  24.592  0.401  0.619  0.346  11654
2  mle_baseline         1          0   50  0.0   test      0.382      0.016     22670  25.639  0.395  0.627  0.352  11849
3  mle_baseline         1          0    1  0.0   test      0.697      0.442     10845  25.639  0.395  0.627  0.352  11849
4  mle_baseline         1          0    1  0.9  valid      0.368      0.014     25574  24.592  0.401  0.619  0.346  11654
5  mle_baseline         1          0    1  0.9   test      0.370      0.016     27275  25.639  0.395  0.627  0.352  11849
6  mle_baseline        10          0    1  0.0  valid      0.726      0.495      9470  24.592  0.401  0.619  0.346  11654
7  mle_baseline        10          0    1  0.0   test      0.740      0.523      9530  25.639  0.395  0.627  0.352  11849
8  mle_baseline        10          4    1  0.0  valid      0.505      0.000     13350  24.592  0.401  0.619  0.346  11654
9  mle_baseline        10          4    1  0.0   test      0.511      0.000     14158  25.639  0.395  0.627  0.352  11849



MODEL: token_level_ul

       model_name beam size beam block topk topp  split  seq-rep-1  seq-rep-4  uniq-seq     ppl    acc    rep   wrep   uniq
0  token_level_ul         1          0   50  0.0  valid      0.303      0.007     22861  25.624  0.396  0.569  0.305  12462
1  token_level_ul         1          0    1  0.0  valid      0.584      0.274     12630  25.624  0.396  0.569  0.305  12462
2  token_level_ul         1          0   50  0.0   test      0.304      0.007     24476  26.910  0.390  0.577  0.311  12728
3  token_level_ul         1          0    1  0.0   test      0.586      0.283     13195  26.910  0.390  0.577  0.311  12728
4  token_level_ul         1          0    1  0.9  valid      0.279      0.005     28859  25.624  0.396  0.569  0.305  12462
5  token_level_ul         1          0    1  0.9   test      0.280      0.005     31325  26.910  0.390  0.577  0.311  12728
6  token_level_ul        10          0    1  0.0  valid      0.615      0.327     11225  25.624  0.396  0.569  0.305  12462
7  token_level_ul        10          0    1  0.0   test      0.619      0.336     11753  26.910  0.390  0.577  0.311  12728
8  token_level_ul        10          4    1  0.0  valid      0.433      0.000     14622  25.624  0.396  0.569  0.305  12462
9  token_level_ul        10          4    1  0.0   test      0.437      0.000     15386  26.910  0.390  0.577  0.311  12728



MODEL: seq_level_ul_mle

         model_name beam size beam block topk topp  split  seq-rep-1  seq-rep-4  uniq-seq     ppl    acc    rep   wrep   uniq
0  seq_level_ul_mle         1          0   50  0.0  valid      0.305  1.000e-03     23169  24.284  0.406  0.603  0.329  12355
1  seq_level_ul_mle         1          0   50  0.0   test      0.307  1.000e-03     24946  25.416  0.399  0.609  0.335  12779
2  seq_level_ul_mle         1          0    1  0.0  valid      0.507  1.306e-01     12663  24.284  0.406  0.603  0.329  12355
3  seq_level_ul_mle         1          0    1  0.0   test      0.514  1.369e-01     13144  25.416  0.399  0.609  0.335  12779
4  seq_level_ul_mle         1          0    1  0.9  valid      0.290  6.000e-04     31012  24.284  0.406  0.603  0.329  12355
5  seq_level_ul_mle         1          0    1  0.9   test      0.294  9.000e-04     33926  25.416  0.399  0.609  0.335  12779
6  seq_level_ul_mle        10          0    1  0.0  valid      0.374  1.830e-02     16817  24.284  0.406  0.603  0.329  12355
7  seq_level_ul_mle        10          0    1  0.0   test      0.376  1.910e-02     18352  25.416  0.399  0.609  0.335  12779
8  seq_level_ul_mle        10          4    1  0.0  valid      0.356  0.000e+00     16898  24.284  0.406  0.603  0.329  12355
9  seq_level_ul_mle        10          4    1  0.0   test      0.358  0.000e+00     18432  25.416  0.399  0.609  0.335  12779



MODEL: seq_level_ul_token_level_ul

                    model_name beam size beam block topk topp  split  seq-rep-1  seq-rep-4  uniq-seq     ppl    acc    rep   wrep   uniq
0  seq_level_ul_token_level_ul         1          0   50  0.0  valid      0.254  5.000e-04     24253  25.375  0.401  0.551  0.287  13375
1  seq_level_ul_token_level_ul         1          0   50  0.0   test      0.257  6.000e-04     25997  26.718  0.395  0.559  0.293  13759
2  seq_level_ul_token_level_ul         1          0    1  0.0  valid      0.428  5.190e-02     14845  25.375  0.401  0.551  0.287  13375
3  seq_level_ul_token_level_ul         1          0    1  0.0   test      0.438  5.850e-02     15428  26.718  0.395  0.559  0.293  13759
4  seq_level_ul_token_level_ul         1          0    1  0.9  valid      0.233  3.000e-04     32011  25.375  0.401  0.551  0.287  13375
5  seq_level_ul_token_level_ul         1          0    1  0.9   test      0.234  3.000e-04     34824  26.718  0.395  0.559  0.293  13759
6  seq_level_ul_token_level_ul        10          0    1  0.0  valid      0.335  1.310e-02     17562  25.375  0.401  0.551  0.287  13375
7  seq_level_ul_token_level_ul        10          0    1  0.0   test      0.338  1.350e-02     19151  26.718  0.395  0.559  0.293  13759
8  seq_level_ul_token_level_ul        10          4    1  0.0  valid      0.322  0.000e+00     17792  25.375  0.401  0.551  0.287  13375
9  seq_level_ul_token_level_ul        10          4    1  0.0   test      0.326  0.000e+00     19439  26.718  0.395  0.559  0.293  13759
```

## Finetuning GPT-2
We also provide a script for sequence-level and maximum-likelihood fine-tuning a GPT-2 model from the [pytorch transformers](https://github.com/huggingface/pytorch-transformers) library. 

Install (we used version `1.1.0`):
```bash
pip install pytorch-transformers
```

We will again assume that you are in the `fairseq` base directory:
```bash
cd $FAIRSEQ_DIR
```

Download and unpack the BPE-tokenized WikiText:
```bash
wget https://dl.fbaipublicfiles.com/unlikelihood/wikitext-103-bpe_v0.tar.gz
tar -xzvf wikitext-103-bpe_v0.tar.gz
mv wikitext-103-bpe_v0 data-bin/
```

#### Sequence-level finetuning
```bash
python fairseq/custom/gpt2/run_gpt2.py  \
    --data-base ./data-bin/wikitext-103-bpe_v0 \
    --output-dir ./checkpoint/gpt2/seq_tune \
    --eval-split valid \
    --mode train
```

#### MLE-tuning
```bash
python fairseq/custom/gpt2/run_gpt2.py  \
    --data-base ./data-bin/wikitext-103-bpe_v0 \
    --output-dir ./checkpoint/gpt2/mle_tune \
    --eval-split valid \
    --train-n-steps 20000 \
    --validate-every 1000 \
    --sequence-tune-rate 0.0 \
    --mode train
```

#### Sequence-level finetuning after MLE-tuning
```
python fairseq/custom/gpt2/run_gpt2.py  \
    --data-base ./data-bin/wikitext-103-bpe_v0 \
    --output-dir ./checkpoint/gpt2/seq_mle_tune \
    --eval-split valid \
    --model-load-dir ./checkpoint/gpt2/mle_tune/best \
    --mode train
```

#### Evaluation
```
python fairseq/custom/gpt2/run_gpt2.py  \
    --data-base ./data-bin/wikitext-103-bpe_v0 \
    --output-dir ./checkpoint/gpt2/seq_mle_tune \
    --eval-split valid \
    --model-load-dir ./checkpoint/gpt2/seq_mle_tune \
    --mode eval-both
```

We used a single Tesla V100 32GB gpu.

## License
unlikelihood_training is CC-BY-NC 4.0 licensed, as found in the LICENSE file.
