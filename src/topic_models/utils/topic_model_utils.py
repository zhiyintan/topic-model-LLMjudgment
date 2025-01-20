import os
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from utils.file_utils import create_path, create_results_file, save_tsv
from utils.data_utils import get_parameter_combinations


def prepare_train_data(train_document_fname, preprocessing, data_source='agris'):
    """
    Load and preprocess training documents.

    Args:
        train_document_fname (str): Path to training documents.
        preprocessing (Preprocessing): Preprocessing object.
        data_source (str): Data source identifier ('agris', '20ng').

    Returns:
        tuple: Preprocessed dataset, tokenized documents, sample documents.
    """
    if data_source == 'agris':
        train_doc = pd.read_csv(train_document_fname, sep='\t', encoding='utf-8')['text'].to_list()
    elif data_source == '20ng':
        train_doc = pd.read_csv(train_document_fname, sep='\t', encoding='utf-8')['text'].to_list()

    dataset = preprocessing.preprocess_raw_documents(train_doc)
    tokenized_docs = [doc.split() for doc in dataset['train_texts']]
    sample_doc = dataset['train_texts']
    return dataset, tokenized_docs, sample_doc


def prepare_test_data(test_document_fname, preprocessing, vocab, data_source='agris'):
    """
    Load and preprocess test documents.

    Args:
        test_document_fname (str): Path to test documents.
        preprocessing (Preprocessing): Preprocessing object.
        vocab (list): Vocabulary list.
        data_source (str): Data source identifier ('agris', '20ng').

    Returns:
        tuple: Tokenized test documents, Bag-of-Words matrix.
    """
    if data_source == 'agris':
        test_doc = pd.read_csv(test_document_fname, sep='\t', encoding='utf-8')['text'].to_list()
    elif data_source == '20ng':
        test_doc = pd.read_csv(test_document_fname, sep='\t', encoding='utf-8')['text'].to_list()

    parsed_new_docs, new_bow = preprocessing.parse(test_doc, vocab=vocab)
    return parsed_new_docs, new_bow


def save_topic_results(top_words, trainset_topic_distribution, model_params, output_type="train"):
    """
    Save topic modeling results (topic words and distributions).

    Args:
        top_words (list): Topic words.
        trainset_topic_distribution (np.ndarray): Topic distribution matrix.
        model_params (dict): Model parameters for naming output files.
        output_type (str): Output type ('train' or 'test').
    """
    base_path = f"results/topic_distribution_{output_type}/"
    params_str = "_".join(str(p) for p in model_params.values())
    topic_words_path = f"results/topic_words/{params_str}.csv"

    # Save topic words
    df_topic_words = pd.DataFrame({
        'ID': range(len(top_words)),
        'Topic words': top_words
    })
    save_tsv(df_topic_words, topic_words_path)

    # Save topic distributions
    np.save(f"{base_path}{params_str}.npy", trainset_topic_distribution)


def save_evaluation_results(eval_file_path, metrics, model_params):
    """
    Save evaluation metrics to a file.

    Args:
        eval_file_path (str): Path to the evaluation results file.
        metrics (tuple): Computed evaluation metrics.
        model_params (dict): Model parameters for saving results.
    """
    with open(eval_file_path, 'a') as f:
        param_str = "\t".join(str(v) for v in model_params.values())
        metric_str = "\t".join(str(m) for m in metrics)
        f.write(f"{param_str}\t{metric_str}\n")


def create_output_paths(model_type):
    """
    Create directories and files for saving results.

    Args:
        model_type (str): Model type identifier (e.g., 'lda', 'prodlda').

    Returns:
        str: Path to the evaluation results file.
    """
    create_path([
        "results/evaluation_result/auto",
        "results/evaluation_result/auto/parameters_iteration",
        "results/topic_words",
        "results/topic_distribution_train",
        "results/topic_distribution_test"
    ])
    eval_file_path = f"results/evaluation_result/auto/{model_type}_parameters_result.csv"
    create_results_file(eval_file_path, "params\tTD\tTC_umass\tTC_cv\tTC_cnpmi\n")
    return eval_file_path
