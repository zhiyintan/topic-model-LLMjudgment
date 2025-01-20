import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
import os
import time


# Topic Metrics Computation
def compute_topic_metrics(top_words, vocab, corpus):
    """
    Compute topic metrics such as Topic Diversity (TD) and Topic Coherence (TC).

    Args:
        top_words (list): List of topic words.
        vocab (list): Vocabulary from the corpus.
        corpus (list): The original text corpus.

    Returns:
        tuple: Metrics (TD, TC_umass, TC_cv, TC_cnpmi).
    """
    try:
        TD = round(compute_topic_diversity(top_words), 4)
    except Exception as e:
        TD = "error"
        print(f"Error computing Topic Diversity: {e}")

    try:
        TC_umass = round(compute_topic_coherence(top_words, vocab, corpus, cv_type='u_mass'), 4)
    except Exception as e:
        TC_umass = "error"
        print(f"Error computing TC_umass: {e}")

    try:
        TC_cv = round(compute_topic_coherence(top_words, vocab, corpus, cv_type='c_v'), 4)
    except Exception as e:
        TC_cv = "error"
        print(f"Error computing TC_cv: {e}")

    try:
        TC_cnpmi = round(compute_topic_coherence(top_words, vocab, corpus, cv_type='c_npmi'), 4)
    except Exception as e:
        TC_cnpmi = "error"
        print(f"Error computing TC_cnpmi: {e}")

    return TD, TC_umass, TC_cv, TC_cnpmi


def compute_topic_diversity(top_words):
    """
    Compute Topic Diversity (TD).

    Args:
        top_words (list): List of topic words.

    Returns:
        float: Topic Diversity value.
    """
    start_time = time.time()

    K = len(top_words)  # Number of topics
    T = len(top_words[0].split())  # Words per topic

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), token_pattern=None)
    counter = vectorizer.fit_transform(top_words).toarray()

    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)

    print(f"Topic Diversity: {TD} (used {time.time() - start_time:.4f} seconds)")
    return TD


def compute_topic_coherence(top_words, vocab, reference_corpus, cv_type='c_v'):
    """
    Compute Topic Coherence (TC) using different methods.

    Args:
        top_words (list): List of topic words.
        vocab (list): Vocabulary.
        reference_corpus (list): Reference corpus.
        cv_type (str): Coherence method ('u_mass', 'c_v', 'c_npmi', etc.).

    Returns:
        float: Topic Coherence score.
    """
    start_time = time.time()

    split_top_words = [topic.split() for topic in top_words]
    num_top_words = len(split_top_words[0])

    for item in split_top_words:
        assert len(item) == num_top_words, f"Inconsistent topic word lengths: {item}"

    split_reference_corpus = [doc.split() for doc in reference_corpus]
    dictionary = Dictionary([voc.split() for voc in vocab])

    cm = CoherenceModel(
        texts=split_reference_corpus, 
        dictionary=dictionary, 
        topics=split_top_words, 
        topn=num_top_words, 
        coherence=cv_type
    )
    score = cm.get_coherence()

    print(f"Topic Coherence {cv_type}: {score} (used {time.time() - start_time:.4f} seconds)")
    return score

# Utility Functions
def save_evaluation_metrics(file_path, test_file, metrics, mode='a'):
    """
    Save evaluation metrics to a CSV file in append mode.

    Args:
        file_path (str): Path to the output file.
        test_file (str): Name of the test file evaluated.
        metrics (dict): Dictionary of evaluation metrics.
        mode (str): File mode, default is append ('a').
    """
    # Prepare the data row
    data_row = {
        "test_file": test_file,
        "llm_model": test_file.split('_')[0],
        "number_of_topics": 100 if int(test_file.split('_')[1]) > 80 else 50,
        "iteration": test_file.split('_')[-1].rsplit('.', 1)[0],
        "mean_precision_at_k": metrics['mean_precision_at_k'],
        "mean_recall_at_k": metrics['mean_recall_at_k'],
        "mean_average_precision": metrics['mean_average_precision'],
        "mean_ndcg_at_k": metrics['mean_ndcg_at_k']
    }

    # Check if the file exists to add a header or not
    write_header = not os.path.exists(file_path)

    # Write the data row
    with open(file_path, mode) as f:
        # Write header if file doesn't exist
        if write_header:
            f.write("test_file\tllm_model\tnumber_of_topics\titeration\tmean_precision_at_k\tmean_recall_at_k\tmean_average_precision\tmean_ndcg_at_k\n")
        # Append the metrics
        f.write(f"{data_row['test_file']}\t"
                f"{data_row['llm_model']}\t{data_row['number_of_topics']}\t{data_row['iteration']}\t"
                f"{data_row['mean_precision_at_k']:.4f}\t{data_row['mean_recall_at_k']:.4f}\t"
                f"{data_row['mean_average_precision']:.4f}\t{data_row['mean_ndcg_at_k']:.4f}\n")
        
def load_and_process_distributions(distribution_path, paperid_2_dataid_dic):
    """
    Compute mean topic distribution for each paper ID.

    Args:
        distribution_path (np.array): Array of topic distributions.
        paperid_2_dataid_dic (dict): Mapping of paper IDs to data IDs.

    Returns:
        tuple: (List of paper IDs, Array of mean distributions for each paper)
    """
    avg_distributions = []
    paper_ids = []

    for paperid, dataid_list in paperid_2_dataid_dic.items():
        paper_distribution = distribution_path[dataid_list]
        mean_distribution = paper_distribution.mean(axis=0)
        avg_distributions.append(mean_distribution)
        paper_ids.append(paperid)

    return paper_ids, np.array(avg_distributions)


def get_top_n_similar(test_distribution, train_distributions, top_n=10):
    """
    Calculate top N similar distributions based on cosine similarity.

    Args:
        test_distribution (np.array): Topic distribution of the test document.
        train_distributions (np.array): Topic distributions of training documents.
        top_n (int): Number of top similar distributions to retrieve.

    Returns:
        tuple: Indices of the top N similar distributions, their distances.
    """
    distances = cdist([test_distribution], train_distributions, metric='cosine')[0]
    top_n_indices = np.argsort(distances)[:top_n]
    return top_n_indices, distances[top_n_indices]



# Recommendation and Evaluation

def recommend_documents(train_paper_ids, train_distributions, test_paper_ids, test_distributions, top_n=10):
    """
    Generate document recommendations for each test paper based on cosine similarity.

    Args:
        train_paper_ids (list): List of train paper IDs.
        train_distributions (np.array): Topic distributions for train papers.
        test_paper_ids (list): List of test paper IDs.
        test_distributions (np.array): Topic distributions for test papers.
        top_n (int): Number of top recommendations to return.

    Returns:
        pd.DataFrame: Recommendations with test paper IDs, ranks, similarity scores, and train paper IDs.
    """
    recommendations = []
    for test_idx, test_dist in enumerate(test_distributions):
        test_paper_id = test_paper_ids[test_idx]
        distances = cdist([test_dist], train_distributions, metric='cosine')[0]
        top_indices = np.argsort(distances)[:top_n]
        top_distances = distances[top_indices]
        for rank, (train_idx, distance) in enumerate(zip(top_indices, top_distances), start=1):
            recommendations.append({
                'test_paper_id': test_paper_id,
                'rank': rank,
                'similarity_score': 1 - distance,
                'train_paper_id': train_paper_ids[train_idx]
            })
    return pd.DataFrame(recommendations)


def evaluate_recommendations(recommendations, label_mapping, k=10):
    """
    Evaluate recommendations based on label overlap, precision, recall, MAP, and NDCG.

    Args:
        recommendations (pd.DataFrame): Recommendations with test and train paper IDs.
        label_mapping (dict): Mapping from paper IDs to label sets.
        k (int): Number of top recommendations to consider.

    Returns:
        dict: Evaluation metrics including precision, recall, MAP, and NDCG.
    """
    precision_at_k = []
    recall_at_k = []
    average_precision = []
    ndcg_at_k = []

    for test_paper_id, group in recommendations.groupby('test_paper_id'):
        test_labels = set(label_mapping.get(test_paper_id, []))
        if not test_labels:
            continue
        recommended_papers = group['train_paper_id'].tolist()
        relevant_labels = set()
        for train_paper_id in recommended_papers[:k]:
            relevant_labels.update(label_mapping.get(train_paper_id, []))

        precision = len(test_labels & relevant_labels) / len(relevant_labels) if relevant_labels else 0
        recall = len(test_labels & relevant_labels) / len(test_labels) if test_labels else 0
        precision_at_k.append(precision)
        recall_at_k.append(recall)

        # Average Precision
        ap = 0
        relevant_found = 0
        for rank, train_paper_id in enumerate(recommended_papers[:k], start=1):
            if test_labels & set(label_mapping.get(train_paper_id, [])):
                relevant_found += 1
                ap += relevant_found / rank
        ap /= len(test_labels) if test_labels else 0
        average_precision.append(ap)

        # NDCG
        ndcg = compute_ndcg(recommended_papers[:k], test_labels, label_mapping, k=k)
        ndcg_at_k.append(ndcg)

    return {
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'average_precision': average_precision,
        'ndcg_at_k': ndcg_at_k,
        'mean_precision_at_k': np.mean(precision_at_k),
        'mean_recall_at_k': np.mean(recall_at_k),
        'mean_average_precision': np.mean(average_precision),
        'mean_ndcg_at_k': np.mean(ndcg_at_k),
    }


def compute_ndcg(recommended_papers, test_labels, label_mapping, k=10):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).

    Args:
        recommended_papers (list): List of recommended train paper IDs (ranked).
        test_labels (set): Relevant labels for the test paper.
        label_mapping (dict): Mapping from paper IDs to their label sets.
        k (int): Number of top recommendations to consider.

    Returns:
        float: NDCG score.
    """
    dcg = compute_dcg(recommended_papers, test_labels, label_mapping, k)
    ideal_relevances = [1] * min(len(test_labels), k)  # Assume perfect ranking
    idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, start=1))
    return dcg / idcg if idcg > 0 else 0


def compute_dcg(recommended_papers, test_labels, label_mapping, k=10):
    """
    Compute Discounted Cumulative Gain (DCG).

    Args:
        recommended_papers (list): List of recommended train paper IDs (ranked).
        test_labels (set): Relevant labels for the test paper.
        label_mapping (dict): Mapping from paper IDs to their label sets.
        k (int): Number of top recommendations to consider.

    Returns:
        float: DCG score.
    """
    dcg = 0
    for rank, train_paper_id in enumerate(recommended_papers[:k], start=1):
        train_labels = set(label_mapping.get(train_paper_id, []))
        relevance = 1 if test_labels & train_labels else 0
        dcg += relevance / np.log2(rank + 1)
    return dcg


