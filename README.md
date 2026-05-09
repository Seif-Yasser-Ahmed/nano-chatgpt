# nano-chatgpt

Reproducing a GPT-2-style language model from scratch, following Andrej
Karpathy's video [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU), and Karpathy's Repo [nanoGPT](https://github.com/karpathy/nanoGPT.git).

This project implements the main pieces from the video: a GPT-2 architecture in
PyTorch, GPT-2 tokenizer usage, next-token training, optimizer setup, learning
rate scheduling, gradient accumulation, mixed precision, FlashAttention,
distributed training, checkpointing, evaluation, sampling, and instruction
fine-tuning.

The final project model is a custom GPT-2-style assistant model with a 1024-token
context window, 12 transformer blocks, 12 attention heads, 768 hidden size, and a
padded vocabulary of 50,304 tokens. The presentation describes it as a
152M-parameter reproduction of GPT-2.

## Table of Contents

- [nano-chatgpt](#nano-chatgpt)
  - [Table of Contents](#table-of-contents)
  - [Project Goal](#project-goal)
  - [What Was Built](#what-was-built)
  - [Repository Structure](#repository-structure)
  - [Installation](#installation)
  - [Architecture](#architecture)
  - [Project Workflow](#project-workflow)
  - [Data Preparation](#data-preparation)
    - [Pretraining Data](#pretraining-data)
    - [Supervised Fine-Tuning Data](#supervised-fine-tuning-data)
  - [Training](#training)
    - [Local Smoke Test](#local-smoke-test)
    - [Cloud / Full Pretraining](#cloud--full-pretraining)
  - [Fine-Tuning](#fine-tuning)
  - [Evaluation](#evaluation)
    - [Evaluate a Single Checkpoint](#evaluate-a-single-checkpoint)
    - [Compare Against OpenAI GPT-2](#compare-against-openai-gpt-2)
  - [Inference](#inference)
    - [Basic Completion CLI](#basic-completion-cli)
    - [Chat CLI](#chat-cli)
  - [Gradio Demo](#gradio-demo)
  - [Results](#results)
    - [HellaSwag Checkpoints](#hellaswag-checkpoints)
    - [Presentation Benchmark Table](#presentation-benchmark-table)
  - [Notes and Limitations](#notes-and-limitations)
  - [License](#license)

## Project Goal

The goal is to reproduce the GPT-2 training pipeline in a compact, educational
codebase:

1. Build the GPT architecture manually in PyTorch.
2. Train it with the same next-token prediction objective used by GPT-2.
3. Optimize the training loop with modern PyTorch features.
4. Pretrain on Edu-FineWeb.
5. Fine-tune on instruction-following conversations (Alpaca_cleaned).
6. Evaluate the model on common language-model benchmarks.
7. Run the trained checkpoint through CLI and Gradio chat interfaces.

The project follows Karpathy's implementation sequence closely, then extends it
with instruction fine-tuning and an attention-visualization demo.

## What Was Built

- GPT-2-style decoder-only Transformer.
- Token and positional embeddings.
- Multi-head causal self-attention.
- FlashAttention through `torch.nn.functional.scaled_dot_product_attention`.
- Optional manual attention path for visualization.
- Transformer MLP blocks, including a modern gated MLP option.
- Weight tying between token embeddings and the language-model head.
- Custom initialization with residual projection scaling.
- GPT-2 checkpoint import support from Hugging Face.
- Pretraining data loader over tokenized binary files.
- Local text-file training path for smaller experiments.
- DDP-aware cloud training loop.
- Gradient accumulation to reach large effective batch sizes.
- AdamW optimizer with decayed and non-decayed parameter groups.
- Warmup plus cosine learning-rate schedule.
- Mixed precision with `torch.autocast`.
- Optional `torch.compile` on CUDA.
- Checkpoint save and resume.
- Supervised fine-tuning on ChatML-style data.
- HellaSwag evaluation through `lm-eval`.
- Baseline comparison against OpenAI GPT-2.
- CLI completion and chat scripts.
- Gradio web demo with BertViz attention display.

## Repository Structure

```text
nano-chatgpt/
|-- main.py                         # Simple prompt/file completion CLI
|-- requirements.txt                # Core Python dependencies
|-- README.md                       # Project documentation
|-- data/
|   |-- edu_fineweb.py              # FineWeb-Edu download/tokenization
|   |-- prepare_alpaca.py           # Alpaca SFT preprocessing
|   `-- prepare_sft.py              # OpenHermes-style SFT preprocessing
|-- src/
|   |-- gpt.py                      # GPT model, generation, checkpoint loading
|   |-- gpt_config.py               # GPTConfig dataclass
|   |-- block.py                    # Transformer block
|   |-- attention.py                # Causal attention and FlashAttention
|   |-- mlp.py                      # GPT MLP and gated MLP
|   |-- data_loader.py              # Binary/text data loader
|   |-- train_local.py              # Local training script
|   |-- train_cloud.py              # Pretraining script
|   |-- sft_cloud.py                # Supervised fine-tuning script
|   |-- chat.py                     # Chat CLI for fine-tuned checkpoints
|   |-- app.py                      # Gradio chat + attention visualization
|   `-- evaluate/
|       |-- evaluate.py             # Evaluate custom checkpoint
|       `-- compare.py              # Compare custom checkpoint to GPT-2
|-- out_gpt2/
|   |-- 10B/                        # Saved 10B-token run checkpoints/results
|   |-- sampled/                    # Generated samples during training
|   `-- outputs_csv/                # Collected samples
`-- test_data/
    `-- input.txt                   # Small local text training data
```

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some scripts import modules from `src` directly. From the project root, set:

```powershell
$env:PYTHONPATH = ".;src"
```

Optional packages for the web demo:

```powershell
pip install gradio bertviz
```

## Architecture

The model is a GPT-2-style decoder-only Transformer:

| Component | Project setting |
| --- | --- |
| Context length | 1024 tokens |
| Layers | 12 |
| Attention heads | 12 |
| Embedding size | 768 |
| Vocabulary | 50,304 padded GPT-2-tokenizer vocabulary |
| Objective | Next-token prediction |
| Main checkpoint size | About 152M parameters |

The presentation explains the model flow as:

1. Text is tokenized into GPT-2 token IDs.
2. Tokens are represented as sparse one-hot indices.
3. Token IDs select rows from an embedding table.
4. Positional embeddings are added to token embeddings.
5. The sequence passes through 12 Transformer blocks.
6. The final hidden states are projected to vocabulary logits.
7. Cross-entropy loss trains the model to predict the next token.

Parameter distribution from the presentation:

| Parameter group | Approximate share |
| --- | ---: |
| MLP / feed-forward networks | 55% |
| Token + positional embeddings | 25% |
| Attention QKV + projection | 18% |
| LayerNorm | 1% |

## Project Workflow

The full project was completed in these stages:

1. Implemented the GPT-2 module in PyTorch.
2. Added token embeddings, positional embeddings, transformer blocks, final
   LayerNorm, and language-model head.
3. Added GPT-2-compatible initialization.
4. Added weight tying between `wte` and `lm_head`.
5. Implemented forward pass and cross-entropy loss.
6. Added sampling and text generation.
7. Added GPT-2 tokenizer support through `tiktoken`.
8. Added Hugging Face GPT-2 weight loading for architecture checking.
9. Built a lightweight data loader for raw text and binary token files.
10. Added optimizer parameter grouping for AdamW.
11. Added learning-rate warmup and cosine decay.
12. Added gradient accumulation for large effective batch size.
13. Added mixed precision, TF32 precision, and CUDA compile support.
14. Replaced manual attention with FlashAttention for fast training.
15. Padded the vocabulary from 50,257 to 50,304 for efficient GPU shapes.
16. Added DDP support for multi-GPU/cloud training.
17. Prepared FineWeb-Edu data for pretraining.
18. Ran pretraining on a 10B-token educational dataset sample.
19. Saved intermediate checkpoints and generated samples during training.
20. Prepared instruction-following data in ChatML format.
21. Fine-tuned the pretrained model on SFT data.
22. Evaluated checkpoints against GPT-2 using HellaSwag.
23. Added CLI inference, chat inference, and a Gradio demo.
24. Added attention visualization through the manual attention path.

## Data Preparation

### Pretraining Data

The pretraining pipeline uses
[HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
specifically the `sample-10BT` split.

Prepare approximately 1B tokens:

```powershell
python data/edu_fineweb.py --tokens 1000000000
```

Prepare approximately 10B tokens:

```powershell
python data/edu_fineweb.py --tokens 10000000000
```

This script:

- downloads FineWeb-Edu,
- selects the requested number of rows,
- splits into train and validation data,
- tokenizes with the GPT-2 tokenizer,
- appends the end-of-text token,
- writes `data/train.bin` and `data/val.bin` as `uint16` memmaps.

The presentation reports the 10B-token dataset as about 28.5 GB of data.

### Supervised Fine-Tuning Data

Two SFT preprocessing scripts are included.

For Alpaca:

```powershell
python data/prepare_alpaca.py
```

This downloads `yahma/alpaca-cleaned`, converts instruction/input/output rows to
ChatML-style conversations, pads examples to 1024 tokens, and writes:

```text
data/sft_x.bin
data/sft_y.bin
```

For an OpenHermes-style JSONL file:

```powershell
python data/prepare_sft.py
```

Expected input:

```text
openhermes_155M_sft_100k.jsonl
```

The SFT format uses special token IDs:

| Token | ID |
| --- | ---: |
| `<\|im_start\|>` | 50257 |
| `<\|im_end\|>` | 50258 |
| `newline` | 198 |

User and system tokens are masked with `-100`, so the loss only trains the model
on assistant responses and stop tokens.

The presentation reports about 52K fine-tuning samples.

## Training

### Local Smoke Test

For a small local run on `test_data/input.txt`:

```powershell
python src/train_local.py
```

This is useful for verifying the architecture and training loop before launching
a larger run.

### Cloud / Full Pretraining

Run pretraining from tokenized `data/train.bin` and `data/val.bin`:

```powershell
python src/train_cloud.py `
  --out_dir out_gpt2 `
  --total_batch_size 524288 `
  --B 64 `
  --T 1024 `
  --max_lr 6e-4 `
  --warmup_steps 715 `
  --max_steps 19073 `
  --eval_interval 100 `
  --eval_iters 20
```

Resume from a checkpoint:

```powershell
python src/train_cloud.py `
  --resume_ckpt out_gpt2/10B/checkpoint_9100/ckpt_09100.pt
```

Important training details:

- Effective batch size: 524,288 tokens.
- Sequence length: 1024.
- Optimizer: AdamW.
- Betas: `(0.9, 0.95)`.
- Weight decay: `0.1` for matrix-like parameters.
- Gradient clipping: `1.0`.
- Learning rate: warmup followed by cosine decay.
- Precision: `bfloat16` autocast when available.
- CUDA speedups: TF32, `torch.compile`, and FlashAttention.
- Distributed training: PyTorch DDP when launched with `torchrun`.

Example DDP launch:

```powershell
torchrun --standalone --nproc_per_node=8 src/train_cloud.py
```

The presentation reports training on an H200 GPU with 141 GB HBM3e memory,
roughly 15 hours of training, at about 150 USD of compute.

## Fine-Tuning

After pretraining, run supervised fine-tuning on `data/sft_x.bin` and
`data/sft_y.bin`:

```powershell
python src/sft_cloud.py `
  --out_dir out_gpt2 `
  --resume_ckpt out_gpt2/10B/checkpoint_19072/ckpt_19072.pt `
  --max_steps 22000
```

The presentation records:

- pretraining shown from steps `9301` to `19072`,
- earlier steps `0` to `9300` were not saved in the shown run,
- fine-tuning from steps `19073` to `21999`.

The fine-tuning script uses the same training loop style as pretraining, but the
data loader uses the `sft` split. During SFT, it periodically samples from a
ChatML-formatted assistant prompt.

## Evaluation

### Evaluate a Single Checkpoint

```powershell
python src/evaluate/evaluate.py `
  --ckpt out_gpt2/10B/finetuned/ckpt_21999.pt `
  --batch_size 8
```

### Compare Against OpenAI GPT-2

```powershell
python src/evaluate/compare.py `
  --ckpt out_gpt2/10B/finetuned/ckpt_21999.pt `
  --out_dir out_gpt2/10B/finetuned/evaluation_results `
  --batch_size 8
```

The comparison script:

1. Evaluates OpenAI GPT-2 through `lm-eval`.
2. Loads the custom checkpoint.
3. Wraps the custom model in a Hugging Face-compatible interface.
4. Evaluates the custom model on HellaSwag.
5. Saves text and JSON reports.

## Inference

### Basic Completion CLI

```powershell
python main.py `
  --ckpt out_gpt2/10B/finetuned/ckpt_21999.pt `
  --prompt "The future of artificial intelligence is" `
  --max_new_tokens 80 `
  --temperature 0.8 `
  --top_k 50
```

Use a file with one prompt per line:

```powershell
python main.py `
  --ckpt out_gpt2/10B/finetuned/ckpt_21999.pt `
  --file prompts.txt
```

### Chat CLI

```powershell
python src/chat.py `
  --model_path out_gpt2/10B/finetuned/ckpt_21999.pt `
  --max_tokens 256 `
  --temp 0.7 `
  --top_k 40
```

The chat CLI wraps each prompt in ChatML-style system, user, and assistant
messages before generation.

## Gradio Demo

Run the web app:

```powershell
python src/app.py
```

The app:

- loads the final fine-tuned checkpoint,
- runs chat generation,
- uses the attention-visualization path,
- displays a BertViz attention view in the browser,
- launches on `localhost:7860`.

Default checkpoint path in the app:

```text
out_gpt2/10B/finetuned/ckpt_21999.pt
```

If the checkpoint is somewhere else, update the path in `src/app.py` before
launching.

## Results

### HellaSwag Checkpoints

Saved JSON results show the custom model improving over the OpenAI GPT-2
baseline on HellaSwag normalized accuracy:

| Checkpoint | Custom accuracy | Custom normalized accuracy | GPT-2 normalized baseline |
| --- | ---: | ---: | ---: |
| Step 9,100 | 28.94% | 31.23% | 31.14% |
| Step 17,000 | 29.86% | 33.35% | 31.14% |
| Step 19,072 | 30.27% | 33.57% | 31.14% |
| Fine-tuned step 21,999 | 30.27% | 33.57% | 31.14% |

### Presentation Benchmark Table

The presentation includes broader benchmark results:

| Benchmark | Metric | OpenAI GPT-2 | Custom pretrained | Custom fine-tuned |
| --- | --- | ---: | ---: | ---: |
| LAMBADA | Accuracy | 32.56% | 26.68% | 26.16% |
| Winogrande | Accuracy | 51.62% | 50.91% | 50.51% |
| PIQA | Acc_Norm | 62.51% | 63.44% | 63.93% |
| ARC_Easy | Acc_Norm | 39.48% | 47.60% | 47.14% |
| HellaSwag | Acc_Norm | 31.14% | 33.57% | 33.57% |

Presentation conclusions:

- Modern educational data improved factual and physical reasoning benchmarks.
- Instruction fine-tuning improved assistant behavior but slightly reduced some
  raw completion metrics.
- At this parameter scale, the model shows trade-offs between benchmark breadth,
  narrative prediction, and instruction-following behavior.
- The data distribution strongly controls final model behavior.


## Notes and Limitations

- Large checkpoints and generated samples are stored under `out_gpt2/`.
- `requirements.txt` contains the core training and evaluation dependencies.
  The Gradio demo also needs `gradio` and `bertviz`.
- The data preparation scripts download datasets from Hugging Face, so they need
  internet access.
- The full 10B-token run requires substantial GPU memory and time.
- Some scripts assume they are run from the repository root with `PYTHONPATH`
  including both `.` and `src`.
- The model is educational and experimental. It is not aligned or safety-tested
  like production chat models.

## License

This repository includes an MIT license in `LICENSE`.
