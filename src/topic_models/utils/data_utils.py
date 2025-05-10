import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from itertools import product
import re
import string
punctuation = ''.join(set(string.punctuation) - set("'"))

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


class RawDataset:
    def __init__(self, docs, data_type, preprocessing, batch_size=200, device='cpu', as_tensor=True):
        # Preprocess the documents
        processed_data = preprocessing.preprocess(docs, data_type)
        self.texts = processed_data['texts']
        self.bow = processed_data['bow']
        self.vocab = processed_data['vocab']
        self.vocab_size = len(self.vocab)
        self.batch_size = batch_size

        # Convert BoW data to tensor if needed
        if as_tensor:
            self.train_data = torch.from_numpy(self.bow).float().to(device)
            self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
            self.test_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=False)

    def get_partitioned_corpus(self, use_validation=False):
        if use_validation:
            train_corpus = [text.split(' ') for text in self.texts]
            val_corpus = train_corpus[:10]
            test_corpus = train_corpus[:10]
            return train_corpus, val_corpus, test_corpus
    def get_vocabulary(self):
        return self.vocab

class Tokenizer:
    def __init__(self, stopwords, min_length):
        self.stopword_set = get_stopwords_set(stopwords)
        self.min_length = min_length
        self.replace = re.compile('[%s]' % re.escape(punctuation))  # Replace punctuation with spaces

    def clean_text(self, text):
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = text.lower()  # Convert to lowercase
        text = self.replace.sub(' ', text)  # Replace punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def tokenize(self, text):
        text = self.clean_text(text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopword_set and len(t) >= self.min_length]
        return tokens


class Preprocessing:
    def __init__(self, stopwords='English', min_doc_count=0, max_doc_freq=1.0, min_length=3, vocab_size=None):
        self.min_doc_count = min_doc_count
        self.max_doc_freq = max_doc_freq
        self.min_length = min_length
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(stopwords, min_length)

    def parse(self, texts, vocab):
        vocab_set = set(vocab)
        parsed_texts = [' '.join([t for t in self.tokenizer.tokenize(text) if t in vocab_set]) for text in texts]
        vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split(), token_pattern=None)
        bow = vectorizer.fit_transform(parsed_texts).toarray()
        return parsed_texts, bow

    def preprocess(self, raw_texts, data_type):
        word_counts = Counter()
        doc_counts = Counter()

        for text in tqdm(raw_texts, desc=f"Processing {data_type} texts"):
            tokens = self.tokenizer.tokenize(text)
            word_counts.update(tokens)
            doc_counts.update(set(tokens))

        # Filter vocabulary
        words, doc_freqs = zip(*doc_counts.items())
        vocab = [word for i, word in enumerate(words) if doc_freqs[i] >= self.min_doc_count]

        if self.vocab_size:
            vocab = vocab[:self.vocab_size]

        vocab.sort()
        parsed_texts, bow = self.parse(raw_texts, vocab)

        return {
            'vocab': vocab,
            'texts': parsed_texts,
            'bow': bow,
        }


def get_stopwords_set(stopwords='English'):
    if stopwords == 'English':
        from gensim.parsing.preprocessing import STOPWORDS as stopwords

    elif stopwords == 'agris':
        df = pd.read_csv("./data/raw/agris/get_stopword/agris_stopword_p.csv", sep='\t', encoding='utf-8')
        agris_stopwords = df['text'].tolist()
        from gensim.parsing.preprocessing import STOPWORDS as stopwords_english
        stopwords = stopwords_english.union(set(agris_stopwords))
    
    elif stopwords == 'tweets_ny':
        df = pd.read_csv("./data/raw/tweets_ny/get_stopword/tweets_ny_stopword_p.csv", sep='\t', encoding='utf-8')
        tweets_ny_stopwords = df['text'].tolist()
        from gensim.parsing.preprocessing import STOPWORDS as stopwords_english
        stopwords = stopwords_english.union(set(tweets_ny_stopwords))

    '''elif stopwords in ['mallet', 'snowball']:
        from topmost.data import download_dataset
        download_dataset('stopwords', cache_path='./')
        path = f'./stopwords/{stopwords}_stopwords.txt'
        stopwords = file_utils.read_text(path)'''
    
    stopword_set = frozenset(stopwords)

    return stopword_set


def get_parameter_combinations(model_type="combinedtm"):
    """
    Generate parameter combinations for topic modeling.
    Args: model_type (str): The type of model ('lda', 'prodlda', 'combinedtm', etc.).
    Returns: list: A list of parameter combinations.
    """
    # best paramiters combination 
    #lda-20ng: 
    #    K=50: alpha:0.5, eta:0.01
    #    K=100: alpha:0.5, eta:0.01
    #lda-agris: 
    #    K=50: alpha:0.5, eta:0.5
    #    K=100: alpha:0.5, eta:0.5
    #lda-tweet_ny: 
    #    K=50: alpha:0.5, eta:0.1
    #    K=100: alpha:0.01, eta:0.01
    if model_type == "lda":
        num_topics_list = [50, 100] #[50, 100]
        passes_list = [5]
        alpha_list = [0.01, 0.1, 0.5] # 'symmetric', 'asymmetric', 0.01, 0.1, 0.5
        eta_list = [0.01, 0.1, 0.5] # 'symmetric', 0.01, 0.1, 0.5
        iterations_list = [100]
        chunksize_list = [5000]
        minimum_probability_list = [0]
        decay_list = [0.5]
        random_state_list = [1] #list(range(10))
        eval_every_list = [10]
        offset_list = [1.0]
        return list(
            product(num_topics_list, passes_list, alpha_list, eta_list,
                    iterations_list, chunksize_list, minimum_probability_list,
                    decay_list, random_state_list, eval_every_list, offset_list)
        )
    
    # best paramiters combination
    #prodlda-20ng: 
    #    K=50: hidden_size:200, dropout:0.1, learning_rate:1e-3, epochs:40
    #    K=100: hidden_size:200, dropout:0.1, learning_rate:5e-3, epochs:40
    #prodlda-agris: 
    #    K=50: hidden_size:200, dropout:0.1, learning_rate:5e-3, epochs:40
    #    K=100: hidden_size:200, dropout:0.1, learning_rate:5e-3, epochs:40
    #prodlda-tweet_ny: 
    #    K=50: hidden_size:200, dropout:0.1, learning_rate:5e-3, epochs:40
    #    K=100: hidden_size:200, dropout:0.5, learning_rate:5e-3, epochs:40
    
    elif model_type == "prodlda":
        num_topics_list = [100] # 50, 100
        hidden_size_list = [200] # 100, 200, 400
        dropout_list = [0.5] # 0.1, 0.3, 0.5
        learning_rate_list = [5e-3] # 1e-3, 2e-3, 5e-3
        epochs_list = [40] # 20, 40
        return list(product(num_topics_list, hidden_size_list, dropout_list, learning_rate_list, epochs_list))

    # best paramiters combination
    #combinedtm-20ng
    #    K=50: hidden_size:200, dropout:0.1, learning_rate:5e-3, epochs:40
    #    K=100: hidden_size:200, dropout:0.1, learning_rate:5e-3, epochs:40
    #combinedtm-agris
    #    K=50: hidden_size:200, dropout:0.1, learning_rate:5e-3, epochs:40
    #    K=100: hidden_size:200, dropout:0.1, learning_rate:2e-3, epochs:40
    #combinedtm-tweet_ny
    #    K=50: hidden_size:200, dropout:0.1, learning_rate:5e-3, epochs:40
    #    K=100: hidden_size:200, dropout:0.3, learning_rate:5e-3, epochs:40
    elif model_type == "combinedtm":
        num_topics_list = [50] #50, 100
        hidden_size_list = [200] # 100, 200, 400
        dropout_list = [0.1] # 0.1, 0.3, 0.5
        learning_rate_list = [5e-3] # 1e-3, 2e-3, 5e-3
        epochs_list = [40] # 20, 40
        return list(product(num_topics_list, hidden_size_list, dropout_list, learning_rate_list, epochs_list))

    # best paramiters combination
    #bertopic-20ng
    #    K=50: n_neighbors:10, n_components:5, min_cluster_size:25, diversity:0.3
    #    K=100: n_neighbors:10, n_components:5, min_cluster_size:13, diversity:0.3
    #bertopic-agris
    #    K=50: n_neighbors:10, n_components:5, min_cluster_size:680, diversity:0.3
    #    K=100: n_neighbors:10, n_components:5, min_cluster_size:370, diversity:0.3
    #bertopic-tweet_ny
    #    K=50: n_neighbors:25, n_components:20, min_cluster_size:20, diversity:0.3
    #    K=100: n_neighbors:35, n_components:25, min_cluster_size:9, diversity:0.5
    elif model_type == "bertopic":
        n_neighbors_list = [10] # 5, 10, 30, 50
        n_components_list = [5] # 5, 10, 15, 20
        min_cluster_size_list = [690] # 10, 15, 20, 25  # 670, 660, 650, 370, 360, 350
        diversity_list = [0.3] # 0.3, 0.5, 0.7
        return list(product(n_neighbors_list, n_components_list, min_cluster_size_list, diversity_list))

    else:
        raise ValueError("Invalid model_type. Choose a valid option.")


def pair_paperid_to_dataid(df):
    """
    Map paper IDs to their corresponding data IDs.

    Args:
        df (pd.DataFrame): Dataframe containing 'id' column.

    Returns:
        dict: Mapping of paper IDs to lists of data IDs.
    """
    paperid_2_dataid_dic = {}
    for dataid in range(len(df)):
        paperid = df.loc[dataid, 'id']
        if paperid not in paperid_2_dataid_dic:
            paperid_2_dataid_dic[paperid] = []
        paperid_2_dataid_dic[paperid].append(dataid)
    return paperid_2_dataid_dic


def nwd_H_J_w_csr(n_wd):
    """
    Calculate the conditional entropy H({d}|w) quantifying the importance of each word.
    Optimized for sparse matrices.

    Args:
        n_wd (csr_matrix): Sparse matrix of word-document counts (VxD).

    Returns:
        np.array: Conditional entropy per word.
    """
    V, D = n_wd.shape

    # Total counts for each word
    N_w = np.array(n_wd.sum(axis=1).transpose())[0]

    # Non-zero data
    n = n_wd.data
    row_n, col_n = n_wd.nonzero()

    # Entropy log for non-zero entries
    n_H = n * np.log2(n)
    ind_sel = np.where(n_H > 0)

    # Sparse matrix of entropy values
    row_H = row_n[ind_sel]
    col_H = col_n[ind_sel]
    data_H = n_H[ind_sel]
    X_H = csr_matrix((data_H, (row_H, col_H)), shape=(V, D))

    # Calculate the entropy
    H_J_w = -np.array(X_H.sum(axis=1).transpose())[0] / N_w + np.log2(N_w)
    return H_J_w