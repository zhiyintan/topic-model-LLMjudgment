import os
import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from src.topic_models.utils.evaluation_utils import compute_topic_metrics
from src.topic_models.utils.file_utils import create_path, create_results_file, save_tsv
from src.topic_models.utils.data_utils import RawDataset, Preprocessing, get_parameter_combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BERTopicData:
    def __init__(self, train_path, test_path, preprocessing_params):
        self.train = self._prepare_data(train_path, preprocessing_params, 'train')
        self.test = self._prepare_data(test_path, preprocessing_params, 'test') if test_path else None

    def _prepare_data(self, path, preprocessing_params, data_type):
        documents_df = pd.read_csv(path, sep='\t', encoding='utf-8')
        documents = documents_df['text'].to_list()
        preprocessing = Preprocessing(**preprocessing_params)
        dataset = RawDataset(documents, data_type, preprocessing, batch_size=5000, device=device, as_tensor=False)
        return {
            "orginal_texts": documents,
            "preprocessed_texts": dataset.texts, 
            "vocab": dataset.vocab}

class BERTopicTrainer:
    def __init__(self, bertopic_data, num_topic_words, parameter_combinations, result_dir, mode='parameters_tuning', print_topic_words=False, run_test=False):
        self.bertopic_data = bertopic_data
        self.result_dir = result_dir
        self.parameter_combinations = parameter_combinations
        self.mode = mode
        self.eval_file_path = self._initialize_results_file()
        self.num_topic_words = num_topic_words
        self.print_topic_words = print_topic_words
        self.run_test = run_test

    def _initialize_results_file(self):
        create_path([
            self.result_dir,
            f"{self.result_dir}/parameters_tuning",
            f"{self.result_dir}/topic_words",
            f"{self.result_dir}/topic_distribution_train",
            f"{self.result_dir}/topic_distribution_test"
        ])
        file_path = f"{self.result_dir}/parameters_tuning/bertopic_results.csv" if self.mode == 'parameters_tuning' else f"{self.result_dir}/bertopic_results.csv"
        create_results_file(file_path, "iteration\tn_neighbors\tn_components\tmin_cluster_size\tdiversity\tnum_topics\tnum_outliers\tTD\tTU\tInverted_RBO\tTC_umass\tTC_cv\tTC_cnpmi\n")
        return file_path

    def train_and_evaluate(self, num_iterations, eval_corpus=None):
        for num_iteration in range(num_iterations):
            for params in tqdm(self.parameter_combinations, desc=f"Training Model with {len(self.parameter_combinations)} Parameter Combinations"):
                self._train_single_iteration(num_iteration, params, eval_corpus)

    def _train_single_iteration(self, num_iteration, params, eval_corpus):
        n_neighbors, n_components, min_cluster_size, diversity = params
        min_samples = min_cluster_size // 10 * 8

        vectorizer_model = CountVectorizer(min_df=1, max_df=1.0)
        embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
        embeddings = embedding_model.encode(self.bertopic_data.train["preprocessed_texts"], show_progress_bar=True)

        umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0, metric='cosine')
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        representation_model = MaximalMarginalRelevance(diversity=diversity)
        ctfidf_model = ClassTfidfTransformer()

        n = 0
        while n < 50:
            topic_model = BERTopic(
                top_n_words=self.num_topic_words,
                nr_topics=None,
                embedding_model=embedding_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                ctfidf_model=ctfidf_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                calculate_probabilities=True,
                verbose=True
            )

            topics, probs = topic_model.fit_transform(self.bertopic_data.train["preprocessed_texts"], embeddings)

            # Get topic words
            topic_words_list = []
            for topic_id, topic_words in topic_model.get_topics().items():
                topic_words = ' '.join([word[0].strip() for word in topic_words]).strip()
                topic_words_list.append(topic_words.strip())
                

            num_topics = len(topic_words_list)

            n += 1
            if num_topics > 3 and num_topics > 0:
                sum_words = sum([len(t.split(' ')) for t in topic_words_list])
                print(f"avg_words:\t{sum_words}")
                if int(sum_words % self.num_topic_words) == 0:
                    break

        if self.print_topic_words:
            for idx, words in enumerate(topic_words_list):
                print(f"Topic {idx-1}: {len(words.split(' '))}, {words}")
        print('\n')

        if self.mode != 'parameters_tuning':
            self._save_topic_words(num_iteration, params, topic_words_list)
            self._save_topic_distributions(probs, num_iteration, params, "train")

        if eval_corpus is None:
            eval_corpus = self.bertopic_data.train["orginal_texts"]

        num_outliers = topic_model.get_topic_freq(-1)
        TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi = compute_topic_metrics(topic_words_list, self.bertopic_data.train["vocab"], eval_corpus)
        self._record_evaluation_results(num_iteration, params, num_topics, num_outliers, TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi)

        if self.run_test and self.bertopic_data.test:
            testset_dataset = self.bertopic_data.test["preprocessed_texts"]
            embeddings = embedding_model.encode(testset_dataset, show_progress_bar=True)
            topics, testset_topic_distribution = topic_model.transform(testset_dataset, embeddings)
            self._save_topic_distributions(testset_topic_distribution, num_iteration, params, "test")

        torch.cuda.empty_cache()

    def _save_topic_words(self, num_iteration, params, topic_words_list):
        topic_words_path = f"{self.result_dir}/topic_words/bertopic_{'_'.join(map(str, params))}_{num_iteration}.csv"
        df_topic_words = pd.DataFrame({
            'ID': range(len(topic_words_list)),
            'Topic words': topic_words_list
        })
        save_tsv(df_topic_words, topic_words_path)

    def _save_topic_distributions(self, topic_distribution, num_iteration, params, data_type):
        distribution_path = f"{self.result_dir}/topic_distribution_{data_type}/bertopic_{'_'.join(map(str, params))}_{num_iteration}.npy"
        print(f"The shape of {data_type} set's topic distribution is ({len(topic_distribution)}, {len(topic_distribution[0])})")
        np.save(distribution_path, topic_distribution)

    def _record_evaluation_results(self, num_iteration, params, num_topics, num_outliers, TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi):
        with open(self.eval_file_path, 'a') as f:
            params_str = ''.join([str(param) + '\t' for param in params]).strip("\t")
            f.write(f"{num_iteration}\t{params_str}\t{num_topics}\t{num_outliers}\t{TD}\t{TU}\t{Inverted_RBO}\t{TC_umass}\t{TC_cv}\t{TC_cnpmi}\n")

if __name__ == "__main__":
    train_path = "../../data/raw/20ng/train.csv"
    test_path = "../../data/raw/20ng/test.csv"
    result_dir = "../../data/topic_model_output/bertopic"
    preprocessing_params = {"vocab_size": 10000, "stopwords": 'English'}
    num_topic_words = 10
    num_iterations = 1
    parameter_combinations = get_parameter_combinations("BERTopic")

    print_topic_words = True
    run_test = True

    bertopic_data = BERTopicData(train_path, test_path, preprocessing_params)
    trainer = BERTopicTrainer(bertopic_data, num_topic_words, parameter_combinations, result_dir, mode=' ', print_topic_words=print_topic_words, run_test=run_test)
    trainer.train_and_evaluate(num_iterations)
