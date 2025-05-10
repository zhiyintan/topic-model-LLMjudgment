import os
import numpy as np
import pandas as pd
import torch
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from tqdm import tqdm

from src.topic_models.utils.evaluation_utils import compute_topic_metrics
from src.topic_models.utils.file_utils import create_path, create_results_file, save_tsv
from src.topic_models.utils.data_utils import RawDataset, Preprocessing, get_parameter_combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LDAData:
    def __init__(self, train_path, test_path, preprocessing_params):
        self.train = self._prepare_data(train_path, preprocessing_params, 'train')
        self.test = self._prepare_data(test_path, preprocessing_params, 'test') if test_path else None

    def _prepare_data(self, path, preprocessing_params, data_type):
        documents_df = pd.read_csv(path, sep='\t', encoding='utf-8')
        documents = documents_df['text'].to_list()
        preprocessing = Preprocessing(**preprocessing_params)
        dataset = RawDataset(documents, data_type, preprocessing, batch_size=2000, device=device, as_tensor=False)
        tokenized_docs = [doc.split() for doc in dataset.texts]
        dictionary = Dictionary(tokenized_docs)
        bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        return {
            "orginal_texts": documents, 
            "preprocessed_texts": dataset.texts,
            "vocab": dataset.vocab,
            "dictionary": dictionary, 
            "bow_corpus": bow_corpus}
    

class LDAModelTrainer:
    def __init__(self, lda_data, num_topic_words, parameter_combinations, result_dir, mode='parameters_tuning', print_topic_words=False, run_test=False):
        self.lda_data = lda_data
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
        file_path = f"{self.result_dir}/parameters_tuning/lda_auto_eval_results.csv" if self.mode == 'parameters_tuning' else f"{self.result_dir}/lda_auto_eval_results.csv"
        create_results_file(file_path, "iteration\tnum_topics\tpasses\talpha\teta\titerations\tchunksize\tminimum_probability\tdecay\trandom_state\teval_every\toffset\tTD\tTU\tInverted_RBO\tTC_umass\tTC_cv\tTC_cnpmi\n")
        return file_path

    def train_and_evaluate(self, num_iterations, eval_corpus=None):
        for num_iteration in range(num_iterations):
            for params in tqdm(self.parameter_combinations, desc=f"Training Model with {len(self.parameter_combinations)} Parameter Combinations"):
                self._train_single_iteration(num_iteration, params, eval_corpus)

    def _train_single_iteration(self, num_iteration, params, eval_corpus):
        (num_topics, passes, alpha, eta, iterations, chunksize, minimum_probability, 
         decay, random_state, eval_every, offset) = params

        topic_model = LdaModel(
            corpus=self.lda_data.train["bow_corpus"],
            id2word=self.lda_data.train["dictionary"],
            num_topics=num_topics,
            passes=passes,
            alpha=alpha, eta=eta, iterations=iterations, chunksize=chunksize, 
            minimum_probability=minimum_probability, decay=decay, eval_every=eval_every, 
            offset=offset, random_state=random_state
        )

        topic_words_list = [[word for word, _ in topic_model.show_topic(topic_id, self.num_topic_words)] 
                          for topic_id in range(num_topics)]
        topic_words_list = [' '.join(words) for words in topic_words_list]

        if self.print_topic_words:
            for idx, words in enumerate(topic_words_list):
                print(f"Topic {idx}: {words}")
            print('\n')

        if self.mode != 'parameters_tuning':
            self._save_topic_words(num_iteration, params, topic_words_list)
            train_bow = self.lda_data.train["bow_corpus"]
            trainset_topic_distribution = self._compute_topic_distribution(topic_model, train_bow, num_topics)
            self._save_topic_distributions(trainset_topic_distribution, num_iteration, params, "train")

        if eval_corpus == None:
            corpus = self.lda_data.train["orginal_texts"]
        elif type(eval_corpus) == int:
            count_sample = eval_corpus
            original_text_list = self.lda_data.train["orginal_texts"]
            step = len(original_text_list) / count_sample
            corpus = [original_text_list[int(i * step)] for i in range(count_sample)]

        TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi = compute_topic_metrics(topic_words_list, self.lda_data.train["vocab"], corpus)
        self._record_evaluation_results(num_iteration, params, TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi)

        # Evaluate test set if provided
        if self.mode != 'parameters_tuning' and self.lda_data.test:
            test_bow = self.lda_data.test["bow_corpus"]
            testset_topic_distribution = self._compute_topic_distribution(topic_model, test_bow, num_topics)
            self._save_topic_distributions(testset_topic_distribution, num_iteration, params, "test")

        torch.cuda.empty_cache()

    def _save_topic_words(self, num_iteration, params, topic_words_list):
        topic_words_path = f"{self.result_dir}/topic_words/lda_{'_'.join(map(str, params))}_{num_iteration}.csv"
        df_topic_words = pd.DataFrame({
            'ID': range(len(topic_words_list)),
            'Topic words': topic_words_list
        })
        save_tsv(df_topic_words, topic_words_path)

    def _save_topic_distributions(self, topic_distribution, num_iteration, params, data_type):
        distribution_path = f"{self.result_dir}/topic_distribution_{data_type}/lda_{'_'.join(map(str, params))}_{num_iteration}.npy"
        print(f"The shape of {data_type} set's topic distribution is ({len(topic_distribution)}, {len(topic_distribution[0])})")
        np.save(distribution_path, topic_distribution)

    @staticmethod
    def _compute_topic_distribution(topic_model, bow_corpus, num_topics):
        distribution = np.zeros((len(bow_corpus), num_topics))
        for i, bow in enumerate(bow_corpus):
            for topic_id, prob in topic_model.get_document_topics(bow, minimum_probability=0.0):
                distribution[i, topic_id] = prob
        return distribution

    def _record_evaluation_results(self, num_iteration, params, TD, TU, Inverted_RBO, TC_umass, TC_cv, TC_cnpmi):
        with open(self.eval_file_path, 'a') as f:
            params = ''.join([str(param)+'\t' for param in params]).strip("\t")
            f.write(f"{num_iteration}\t{params}\t{TD}\t{TU}\t{Inverted_RBO}\t{TC_umass}\t{TC_cv}\t{TC_cnpmi}\n")

if __name__ == "__main__":
    train_path = "../../data/raw/20ng/train.csv"
    test_path = "../../data/raw/20ng/test.csv"
    result_dir = "../../data/topic_model_output/lda"
    preprocessing_params = {"vocab_size": 10000, "stopwords": 'English'}
    num_topic_words = 10; num_iterations = 1
    parameter_combinations = get_parameter_combinations("lda")

    print_topic_words = True
    run_test = True

    lda_data = LDAData(train_path, test_path, preprocessing_params)
    trainer = LDAModelTrainer(lda_data, num_topic_words, parameter_combinations, result_dir, mode='parameters_tuning', print_topic_words=print_topic_words, run_test=run_test)
    #trainer = LDAModelTrainer(lda_data, num_topic_words, parameter_combinations, result_dir, mode=' ', print_topic_words=print_topic_words, run_test=run_test)
    trainer.train_and_evaluate(num_iterations)

    
