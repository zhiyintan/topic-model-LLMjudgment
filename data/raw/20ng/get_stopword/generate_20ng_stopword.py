## import packages

#%load_ext autoreload
#%autoreload 2

import os,sys
import numpy as np
import pandas as pd
from tqdm import tqdm


# display the figure in the notebook
# %matplotlib inline
# import matplotlib.pyplot as plt
# cmap = 'tab10'
# cm = plt.get_cmap(cmap)

## custom packages
src_dir = os.path.join('src')
sys.path.append(src_dir)

from filter_words import run_stopword_statistics
from filter_words import make_stopwords_filter
from filter_words import remove_stopwords_from_list_texts



corpus_name = '20ng'
filename = "../tokenized_train.csv"
#os.path.join(os.pardir,'data','%s_corpus.csv'%(corpus_name))
df = pd.read_csv(filename, sep="\t", encoding='utf-8')
print(df.columns)
list_texts = df['text'].apply(lambda x: x.split()).to_list()


## this is the first doc: list_texts[0]

## path to a manual stopword list (this one is from mallet)
path_stopword_list = "stopword_list_en"
#os.path.join(os.pardir,'data','stopword_list_en')

## number of realizations for the random null model
N_s = 10

## get the statistics
df = run_stopword_statistics(list_texts,N_s=N_s,path_stopword_list=path_stopword_list)

## look at the entries
df.sort_values(by='F',ascending=False).head()

## method-options
method = 'INFOR'
# method = 'BOTTOM'
# method = 'TOP'
# method = 'TFIDF'
# method = 'TFIDF_r'
# method = 'MANUAL'



## remove fraction of tokens
cutoff_type = 'p'
cutoff_val = 0.8

## remove number of types
# cutoff_type = 'n'
# cutoff_val = 10

## remove above a threshold value
# cutoff_type = 't'
# cutoff_val = 1

df_filter = make_stopwords_filter(df,
                                  method = method,
                                  cutoff_type = cutoff_type, 
                                  cutoff_val = cutoff_val, )

df.to_csv("20ng_stopword_original.csv", sep='\t', encoding='utf-8')
df_filter.to_csv("20ng_stopword_p.csv", sep='\t', encoding='utf-8')

## get the list of words from df_filter and get a filtered list_of_texts
list_words_filter = list(df_filter.index)
list_texts_filter = remove_stopwords_from_list_texts(list_texts, list_words_filter)

print('Original text:', list_texts[0])
print('Filtered text:', list_texts_filter[0])
N = sum([ len(doc) for doc in list_texts ])
N_filter = sum([ len(doc) for doc in list_texts_filter ])
print('Remaining fraction of tokens',N_filter/N)