# ai-sec-seminar-experiment

This repository contains the scripts and analysis used for our AI Security seminar project on **cross-model jailbreak transferability**. We replicate the _JailbreakHub / ForbiddenQuestions_ evaluation setup from prior jailbreak literature and test whether in-the-wild jailbreak prompts transfer to newer / differently aligned LLMs.

## Project Overview

We evaluate two conditions:

- **Baseline:** the forbidden question alone
- **Jailbreak:** `jailbreak_prompt + question` (same question)

For each selected question we run:

- **1 baseline** request
- **50 jailbreak** requests (one per sampled jailbreak prompt)

This yields **51 runs per question**.

## Datasets

We use two Hugging Face datasets:

- **Forbidden questions (scenarios + questions):**  
  https://huggingface.co/datasets/walledai/ForbiddenQuestions

- **Jailbreak prompt corpus:**  
  https://huggingface.co/datasets/walledai/JailbreakHub

### Note on “communities”

The original JailbreakHub paper discusses community structure/clusters. In the Hugging Face parquet files we used, we did **not** find community IDs/cluster labels exposed as columns. The available metadata includes fields such as `platform` and `source`. Therefore, our experiments and analysis focus on prompt-level transferability rather than community-level comparisons.

## Experimental Setup

### Scenarios and questions

We evaluate **6 forbidden scenarios**, sampling **10 questions per scenario** (60 questions total):

- Illegal Activity
- Hate Speech
- Malware
- Physical Harm
- Financial Advice
- Fraud

### Jailbreak prompt sampling

From JailbreakHub, we filter prompts where `jailbreak == True` and sample **50 prompts** uniformly at random **without replacement** using a fixed random seed:

- `SEED = 42`

The sampled jailbreak prompts are kept consistent across models to enable fair comparison.

### Input formatting

For jailbreak runs, we send the following combined input to the model:
