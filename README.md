# Novelty Detection via LLM-based Retriever and Benchmark Dataset

This repo contains the code and datasets for the paper:  
**"Harnessing Large Language Models for Scientific Novelty Detection" (In submission)**

## Overview

We present:
- Two ND-tailored benchmark datasets (Marketing + NLP domain)
- An LLM-based knowledge distillation framework for training idea-level retriever
- A novelty detection pipeline combining retrieval-augmented generation (RAG) and LLM scoring

## Installation

```bash
conda create -n novelty python=3.10
conda activate novelty
pip install -r requirements.txt
