# Bridging the Evaluation Gap: Leveraging Large Language Models for Topic Model Evaluation

## Overview

This repository provides a framework for **automated evaluation of dynamically evolving topic taxonomies** in scientific literature using **Large Language Models (LLMs)**. The framework addresses challenges in evaluating topic models, such as static metrics and reliance on expert annotators, by leveraging LLMs to assess key quality dimensions like coherence, diversity, repetitiveness, and topic-document alignment.

By integrating multiple topic modeling techniques and an LLM-based evaluation pipeline, this approach ensures robust, scalable, and interpretable results for a variety of datasets and research needs.

## Key Features

- **Modular Topic Modeling:** Includes support for multiple topic modeling techniques such as LDA, ProdLDA, CombinedTM, and BERTopic.
- **LLM-Based Evaluation:** Offers scalable and dynamic evaluations of topic models using tailored LLM prompts.
- **Customizable Pipelines:** Allows for parameter tuning, evaluation metric customization, and integration with new datasets.
- **Tutorial Jupyter Notebook:** Demonstrates how to preprocess data, run topic models, and interpret the results.

## Usage

### 1. Topic Modeling
The `src/topic_models/` directory contains scripts for different topic modeling techniques:
- **`lda.py`:** Latent Dirichlet Allocation (LDA)
- **`prodlda.py`:** Product of Experts LDA (ProdLDA)
- **`combinedtm.py`:** Combined Topic Model (CombinedTM)
- **`bertopic.py`:** BERTopic

You can find detailed instructions in the **tutorial Jupyter notebook**, which walks through the entire topic modeling process, from data preparation to extracting and interpreting topic distributions.

### 2. LLM-Based Evaluation (Coming Soon)
The scripts for LLM-based evaluation are located in `src/llm_judgement/`:
- **`llm_judgment.py`:** Core evaluation logic.
- **`prompt_templates.py`:** Predefined prompt templates for LLMs.
- **`llm_model.py`:** Helper functions to interact with LLMs.

These scripts will be uploaded soon.
