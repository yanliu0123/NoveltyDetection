# Harnessing Large Language Models for Scientific Novelty Detection

This repository contains the official implementation of **LLM-based Scientific Novelty Detection**, a framework that leverages large language models (LLMs) for benchmark construction, idea-level retriever distillation, and retrieval-augmented novelty detection.

Scientific novelty detection aims to determine whether a research idea is conceptually novel with respect to existing literature. Instead of relying only on surface-level textual similarity, this project focuses on idea-level conceptual alignment between research ideas.

## Overview

The framework consists of three main components:

1. **Benchmark Dataset Construction**  
   We construct novelty detection datasets with topological closure and compactness. Seed papers are collected from specific research domains, and their references are crawled to form a closed corpus. LLMs are then used to summarize each paper into compact idea descriptions.

2. **LLM-based Knowledge Distillation for Idea Retrieval**  
   We generate synthesized non-novel ideas from anchor ideas using LLMs, including rephrased, partial, and incremental ideas. These anchor-synthesized idea pairs are used to fine-tune a lightweight retriever with contrastive learning, aligning the retriever with idea-level similarity rather than surface textual similarity.

3. **RAG-based Novelty Detection**  
   Given a target research idea, the distilled retriever first retrieves top-K conceptually related ideas. Then, an LLM cross-checks the target idea against the retrieved candidates and produces novelty scores. A decision tree classifier is used to make the final Novel / Non-Novel prediction.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── acl/                         # Raw ACL/NLP-domain data
│   │   └── marketing/                   # Raw Marketing-domain data
│   └── processed/
│       └── nc/
│           ├── acl/                     # Processed ACL/NLP novelty detection data
│           └── marketing/               # Processed Marketing novelty detection data
├── scripts/
│   ├── utils/
│   │   ├── baseline.py                  # Baseline novelty detection methods
│   │   ├── config.json                  # Configuration file
│   │   ├── csv_processing.py            # CSV processing utilities
│   │   ├── data_preprocessing.py        # Data preprocessing script
│   │   ├── dataset.py                   # Dataset loading and processing
│   │   ├── embeddings.py                # Embedding generation utilities
│   │   ├── general.py                   # General helper functions
│   │   ├── llm.py                       # LLM calling and prompting utilities
│   │   ├── paper_search.py              # Paper search / retrieval utilities
│   │   └── pdf_processing.py            # PDF parsing and processing
│   ├── retrieval/
│   │   ├── train.py                     # Train idea-level retriever
│   │   └── test.py                      # Evaluate idea retrieval performance
│   └── nc/
│       ├── classifier.py                # Decision tree classifier for novelty detection
│       ├── deepseek_parallel.py         # Parallel LLM-based novelty scoring
│       └── run_novelty_checking_integrated.py  # Integrated novelty checking pipeline
````

## Requirements

* Python 3.8+
* PyTorch
* transformers
* sentence-transformers
* scikit-learn
* numpy
* pandas
* tqdm

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Datasets

We provide two benchmark datasets for scientific novelty detection:

| Dataset   |                      Domain | Seed Papers |     Reference Corpus | Description                                                                  |
| --------- | --------------------------: | ----------: | -------------------: | ---------------------------------------------------------------------------- |
| Marketing |  Social Science / Marketing |         470 | 12,577 unique papers | Papers collected from Journal of Marketing and Journal of Marketing Research |
| NLP / ACL | Natural Language Processing |       3,533 | 32,239 unique papers | Papers collected from recent ACL conferences                                 |

The datasets are designed with:

* **Topological closure**: reference papers of seed papers are included to approximate the prior literature used for novelty judgment.
* **Compactness**: each paper is represented by an LLM-generated idea summary, including its core contribution, hypothesis, and methodology.
* **Synthesized non-novel ideas**: LLM-generated rephrased, partial, and incremental ideas are used for retriever training and novelty detection evaluation.

## Quick Start

### Run the integrated novelty detection pipeline

```bash
python scripts/nc/run_novelty_checking_integrated.py \
    --dataset acl \
    --config scripts/utils/config.json
```

For the Marketing dataset:

```bash
python scripts/nc/run_novelty_checking_integrated.py \
    --dataset marketing \
    --config scripts/utils/config.json
```

## Run Step by Step

### Step 1: Data preprocessing

```bash
python scripts/utils/data_preprocessing.py \
    --dataset acl \
    --input_dir data/raw/acl \
    --output_dir data/processed/nc/acl
```

For Marketing:

```bash
python scripts/utils/data_preprocessing.py \
    --dataset marketing \
    --input_dir data/raw/marketing \
    --output_dir data/processed/nc/marketing
```

### Step 2: Generate / load embeddings

```bash
python scripts/utils/embeddings.py \
    --dataset acl \
    --data_dir data/processed/nc/acl
```

### Step 3: Train the idea-level retriever

```bash
python scripts/retrieval/train.py \
    --dataset acl \
    --data_dir data/processed/nc/acl
```

### Step 4: Evaluate idea retrieval

```bash
python scripts/retrieval/test.py \
    --dataset acl \
    --data_dir data/processed/nc/acl
```

### Step 5: Run LLM-based novelty checking

```bash
python scripts/nc/deepseek_parallel.py \
    --dataset acl \
    --data_dir data/processed/nc/acl
```

### Step 6: Train / evaluate the novelty classifier

```bash
python scripts/nc/classifier.py \
    --dataset acl \
    --data_dir data/processed/nc/acl
```

## Key Hyperparameters

| Parameter       |             Default | Description                                               |
| --------------- | ------------------: | --------------------------------------------------------- |
| `dataset`       |               `acl` | Dataset name, selected from `acl` and `marketing`         |
| `retriever`     |               `bge` | Retriever backbone for idea retrieval                     |
| `learning_rate` |              `2e-5` | Learning rate for retriever fine-tuning                   |
| `batch_size`    |                `16` | Batch size for contrastive retriever training             |
| `temperature`   |              `0.05` | Temperature parameter in contrastive learning             |
| `top_k`         |            `5 / 10` | Number of retrieved ideas for RAG-based novelty detection |
| `llm`           | `deepseek-reasoner` | LLM backbone for novelty checking                         |
| `classifier`    |     `decision_tree` | Classifier used for the final Novel / Non-Novel decision  |

## Method Details

### LLM-based KD Retriever

Given an anchor idea and its LLM-synthesized non-novel variant, the retriever is trained to pull conceptually similar ideas closer while pushing unrelated ideas away.

The training objective follows a contrastive learning formulation:

```text
L = - log exp(sim(f(s_i), f(g_i)) / τ)
        / Σ_j exp(sim(f(s_j), f(g_i)) / τ)
```

where:

* `s_i` is an anchor idea from the novelty corpus,
* `g_i` is a synthesized idea generated by an LLM,
* `f(.)` is the retriever encoder,
* `sim(.)` is cosine similarity,
* `τ` is the temperature.

### RAG-based Novelty Detection

For each target idea, the trained retriever retrieves the top-K most conceptually related ideas. An LLM then compares the target idea against these retrieved candidates and outputs novelty scores. These scores are fed into a supervised decision tree classifier for the final novelty prediction.

## Results

The proposed LLM-KD retriever consistently improves idea retrieval performance across both Marketing and NLP datasets. In the novelty detection task, the RAG-KD method achieves the best overall performance compared with heuristic novelty metrics, LLM-only baselines, and RAG with vanilla retrievers.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{liu2026scientificnovelty,
  title={Harnessing Large Language Models for Scientific Novelty Detection},
  author={Liu, Yan and Yang, Zonglin and Poria, Soujanya and Nguyen, Thanh-Son and Cambria, Erik},
  booktitle={International Conference on Artificial Neural Networks},
  year={2026}
}
```

## Contact

For questions, please contact:

```text
Yan Liu
College of Computing and Data Science
Nanyang Technological University, Singapore
Email: yan010@e.ntu.edu.sg
```
