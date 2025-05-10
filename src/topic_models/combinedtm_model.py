import os
import numpy as np
import pandas as pd
import torch
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

from src.topic_models.utils.evaluation_utils import compute_topic_metrics
from src.topic_models.utils.file_utils import create_path, create_results_file, save_tsv
from src.topic_models.utils.data_utils import RawDataset, Preprocessing, get_parameter_combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CombinedTMData:
    def __init__(self, train_path, test_path, preprocessing_params, embed_model="all-MiniLM-L12-v2"):
        self.qt = TopicModelDataPreparation(embed_model)
        self.train = self._prepare_data(train_path, preprocessing_params, 'train', fit_vectorizer=True)
        self.test = self._prepare_data(test_path, preprocessing_params, 'test', fit_vectorizer=False) if test_path else None

    def _prepare_data(self, path, preprocessing_params, data_type, fit_vectorizer):
        documents_df = pd.read_csv(path, sep='\t', encoding='utf-8')

        documents = documents_df['text'].to_list()
        preprocessing = Preprocessing(**preprocessing_params)
        dataset = RawDataset(documents, data_type, preprocessing, batch_size=2000, device=device, as_tensor=False)
        
        if fit_vectorizer:
            combined_dataset = self.qt.fit(text_for_contextual=documents, text_for_bow=dataset.texts)
        else:
            combined_dataset = self.qt.transform(text_for_contextual=documents, text_for_bow=dataset.texts)
        
        return {
            "orginal_texts": documents,
            "combined_texts": combined_dataset,
            "vocab": dataset.vocab
        }

class CombinedTMTrainer:
    def __init__(self, combinedtm_data, num_topic_words, parameter_combinations, result_dir, mode='parameters_tuning', print_topic_words=False, run_test=False):
        self.combinedtm_data = combinedtm_data
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
        file_path = f"{self.result_dir}/parameters_tuning/combinedtm_results.csv" if self.mode == 'parameters_tuning' else f"{self.result_dir}/combinedtm_results.csv"
        create_results_file(file_path, "iteration\tnum_topics\thidden_size\tdropout\tlearning_rate\tepochs\tTD\tTU\tInverted_RBO\tTC_umass\tTC_cv\tTC_cnpmi\n")
        return file_path

    def train_and_evaluate(self, num_iterations, eval_corpus=None):
        for num_iteration in range(num_iterations):
            for params in self.parameter_combinations:
                self._train_single_iteration(num_iteration, params, eval_corpus)

    def _train_single_iteration(self, num_iteration, params, eval_corpus):
        num_topics, hidden_size, dropout, learning_rate, epochs = params

        topic_model = CombinedTM(
            bow_size=self.combinedtm_data.train["combined_texts"].X_bow.shape[1],
            contextual_size=384,  # Example embedding size for "all-MiniLM-L12-v2"
            #inference_type = "combined",
            n_components=num_topics,
            model_type="prodLDA",
            hidden_sizes=(hidden_size, hidden_size), 
            activation="softplus",
            dropout=dropout, 
            batch_size=min(len(self.combinedtm_data.train["orginal_texts"]), 200),
            lr=learning_rate,
            solver="adam",
            num_epochs=epochs,
            num_data_loader_workers=12
        )

        topic_model.fit(self.combinedtm_data.train["combined_texts"])

        topic_words_list = [
            ' '.join(words) for topic_id, words in topic_model.get_topics(k=self.num_topic_words).items()
        ]

        if self.print_topic_words:
            for idx, words in enumerate(topic_words_list):
                print(f"Topic {idx}: {words}")
            print('\n')

        if self.mode != 'parameters_tuning': 
            self._save_topic_words(num_iteration, params, topic_words_list)
            trainset_topic_distribution = topic_model.get_thetas(self.combinedtm_data.train["combined_texts"])
            self._save_topic_distributions(trainset_topic_distribution, num_iteration, params, "train")

        if eval_corpus == None:
            eval_corpus = self.combinedtm_data.train["orginal_texts"]
        TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi = compute_topic_metrics(topic_words_list, self.combinedtm_data.train["vocab"], eval_corpus)
        self._record_evaluation_results(num_iteration, params, TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi)

        if self.run_test and self.combinedtm_data.test:
            testset_dataset = self.combinedtm_data.test["combined_texts"]
            testset_topic_distribution = topic_model.get_thetas(testset_dataset)
            self._save_topic_distributions(testset_topic_distribution, num_iteration, params, "test")

        torch.cuda.empty_cache()

    def _save_topic_words(self, num_iteration, params, topic_words_list):
        topic_words_path = f"{self.result_dir}/topic_words/combinedtm_{'_'.join(map(str, params))}_{num_iteration}.csv"
        df_topic_words = pd.DataFrame({
            'ID': range(len(topic_words_list)),
            'Topic words': topic_words_list
        })
        save_tsv(df_topic_words, topic_words_path)

    def _save_topic_distributions(self, topic_distribution, num_iteration, params, data_type):
        distribution_path = f"{self.result_dir}/topic_distribution_{data_type}/combinedtm_{'_'.join(map(str, params))}_{num_iteration}.npy"
        print(f"The shape of {data_type} set's topic distribution is ({len(topic_distribution)}, {len(topic_distribution[0])})")
        np.save(distribution_path, topic_distribution)

    def _record_evaluation_results(self, num_iteration, params, TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi):
        with open(self.eval_file_path, 'a') as f:
            params_str = ''.join([str(param) + '\t' for param in params]).strip("\t")
            f.write(f"{num_iteration}\t{params_str}\t{TD}\t{TU}\t{Inverted_RBO}\t{TC_umass}\t{TC_cv}\t{TC_cnpmi}\n")

if __name__ == "__main__":
    train_path = "../../data/raw/20ng/train.csv"
    test_path = "../../data/raw/20ng/test.csv"
    result_dir = "../../data/topic_model_output/combinedtm"
    preprocessing_params = {"vocab_size": 10000, "stopwords": 'English'}
    num_topic_words = 10
    num_iterations = 1
    parameter_combinations = get_parameter_combinations("combinedtm")

    print_topic_words = True
    run_test = True

    combinedtm_data = CombinedTMData(train_path, test_path, preprocessing_params)
    trainer = CombinedTMTrainer(combinedtm_data, num_topic_words, parameter_combinations, result_dir, mode='parameters_tuning', print_topic_words=print_topic_words, run_test=run_test)
    trainer.train_and_evaluate(num_iterations)
