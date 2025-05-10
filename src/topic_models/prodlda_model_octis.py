import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from octis.dataset.dataset import Dataset
from octis.models.pytorchavitm.datasets import BOWDataset
from octis.models.ProdLDA import ProdLDA
from topmost.trainers import BasicTrainer

from src.topic_models.utils.evaluation_utils import compute_topic_metrics
from src.topic_models.utils.file_utils import create_path, create_results_file, save_tsv
from src.topic_models.utils.data_utils import RawDataset, Preprocessing, get_parameter_combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ProdLDAData:
    def __init__(self, train_path, test_path, preprocessing_params):
        self.train = self._prepare_data(train_path, preprocessing_params, 'train')
        self.test = self._prepare_data(test_path, preprocessing_params, 'test') if test_path else None
        
        vectorizer = CountVectorizer(vocabulary=self.train['vocab'], tokenizer=lambda x: x.split(), token_pattern=None)
        idx2token = {v: k for (k, v) in enumerate(self.train['vocab'])}
        X_train = vectorizer.transform(self.train['orginal_texts'])
        X_test = vectorizer.transform(self.test['orginal_texts']) if test_path else None
        self.train_bow = BOWDataset(X_train.toarray(), idx2token)
        self.test_bow = BOWDataset(X_test.toarray(), idx2token) if test_path else None

    def _prepare_data(self, path, preprocessing_params, data_type):
        documents_df = pd.read_csv(path, sep='\t', encoding='utf-8')
        documents = documents_df['text'].to_list()
        preprocessing = Preprocessing(**preprocessing_params)
        dataset = RawDataset(documents, data_type, preprocessing, batch_size=2000, device=device, as_tensor=False, )
        return {
            "orginal_texts": documents,
            "dataset": dataset, 
            "vocab": dataset.vocab
        }

class ProdLDATrainer:
    def __init__(self, prodlda_data, num_topic_words, parameter_combinations, result_dir, mode='parameters_tuning', print_topic_words=False, run_test=False):
        self.prodlda_data = prodlda_data
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
        file_path = f"{self.result_dir}/parameters_tuning/prodlda_auto_eval_results.csv" if self.mode == 'parameters_tuning' else f"{self.result_dir}/prodlda_auto_eval_results.csv"
        create_results_file(file_path, "iteration\tnum_topics\thidden_size\tdropout\tlearning_rate\tepochs\tTD\tTU\tInverted_RBO\tTC_umass\tTC_cv\tTC_cnpmi\n")
        return file_path

    def train_and_evaluate(self, num_iterations, eval_corpus=None):
        for num_iteration in range(num_iterations):
            for params in tqdm(self.parameter_combinations, desc=f"Training Model with {len(self.parameter_combinations)} Parameter Combinations"):
                self._train_single_iteration(num_iteration, params, eval_corpus)

    def _train_single_iteration(self, num_iteration, params, eval_corpus):
        num_topics, hidden_size, dropout, learning_rate, epochs = params

        # Initialize ProdLDA model
        dataset = self.prodlda_data.train["dataset"]
        vocab_size = len(self.prodlda_data.train["vocab"])
        print(f"num_topics:{num_topics},\n hidden_size:{hidden_size},\n dropout:{dropout},\n learning_rate:{learning_rate},\n epochs:{epochs}")
        topic_model = ProdLDA(num_topics=num_topics, 
                              num_neurons=hidden_size, 
                              num_epochs=epochs,
                              dropout=dropout,
                              lr=learning_rate, #2e-3,
                              activation='softplus',solver='adam',batch_size=64,num_layers=2,num_samples=10)
            
        # Train the model
        output = topic_model.train_model(dataset, hyperparameters=None, top_words=self.num_topic_words)
        #topic-word-matrix,topics, topic-document-matrix, test-topic-document-matrix

        topic_words = [' '.join(topic_word_list) for topic_word_list in output['topics']]
        trainset_topic_distribution = np.asarray(output['topic-document-matrix']).T
        print(len(trainset_topic_distribution),len(trainset_topic_distribution[0]))

        if self.print_topic_words:
            for idx, words in enumerate(topic_words):
                print(f"Topic {idx}: {words}")
            print('\n')

        # Save results
        if self.mode != 'parameters_tuning':
            self._save_topic_words(num_iteration, params, topic_words)
            self._save_topic_distributions(trainset_topic_distribution, num_iteration, params, "train")

        # Evaluate metrics
        if eval_corpus == None:
            corpus = self.prodlda_data.train["orginal_texts"]
        elif type(eval_corpus) == int:
            count_sample = eval_corpus
            original_text_list = self.prodlda_data.train["orginal_texts"]
            step = len(original_text_list) / count_sample
            corpus = [original_text_list[int(i * step)] for i in range(count_sample)]

        TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi = compute_topic_metrics(
            topic_words, self.prodlda_data.train["vocab"], corpus
        )
        self._record_evaluation_results(num_iteration, params, TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi)
        
        # Evaluate test set if provided
        if self.mode != 'parameters_tuning' and self.prodlda_data.test:
            print("start to test!!")
            test_dataset = self.prodlda_data.test_bow
            testset_topic_distribution = np.asarray(topic_model.inference(test_dataset)['test-topic-document-matrix']).T
            #testset_topic_distribution = output['test-topic-document-matrix']
            #print(len(testset_topic_distribution), len(testset_topic_distribution[0]))
            self._save_topic_distributions(testset_topic_distribution, num_iteration, params, "test")

        torch.cuda.empty_cache()

    def _save_topic_words(self, num_iteration, params, topic_words):
        topic_words_path = f"{self.result_dir}/topic_words/prodlda_{'_'.join(map(str, params))}_{num_iteration}.csv"
        df_topic_words = pd.DataFrame({
            'ID': range(len(topic_words)),
            'Topic words': topic_words
        })
        save_tsv(df_topic_words, topic_words_path)

    def _save_topic_distributions(self, topic_distribution, num_iteration, params, data_type):
        distribution_path = f"{self.result_dir}/topic_distribution_{data_type}/prodlda_{'_'.join(map(str, params))}_{num_iteration}.npy"
        print(f"The shape of {data_type} set's topic distribution is ({len(topic_distribution)}, {len(topic_distribution[0])})")
        np.save(distribution_path, topic_distribution)

    def _record_evaluation_results(self, num_iteration, params, TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi):
        with open(self.eval_file_path, 'a') as f:
            params_str = ''.join([str(param) + '\t' for param in params]).strip("\t")
            f.write(f"{num_iteration}\t{params_str}\t{TD}\t{TU}\t{Inverted_RBO}\t{TC_umass}\t{TC_cv}\t{TC_cnpmi}\n")

if __name__ == "__main__":
    train_path = "../../data/raw/20ng/train.csv"
    test_path = "../../data/raw/20ng/test.csv"
    result_dir = "../../data/topic_model_output/prodlda"
    preprocessing_params = {"vocab_size": 10000, "stopwords": 'English'}
    num_topic_words = 10
    num_iterations = 1
    parameter_combinations = get_parameter_combinations("ProdLDA")

    print_topic_words = True
    run_test = True

    prodlda_data = ProdLDAData(train_path, test_path, preprocessing_params)
    trainer = ProdLDATrainer(prodlda_data, num_topic_words, parameter_combinations, result_dir, mode=' ', print_topic_words=print_topic_words, run_test=run_test)
    trainer.train_and_evaluate(num_iterations)
