import os
import re
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

import spacy
nlp = spacy.load("en_core_web_sm")



def split_hashtags(text):
    # Expand hashtags using CamelCase splitting
    def expand(match):
        hashtag = match.group(0)[1:]  # remove "#"
        # Insert space before each capital letter except the first
        return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', hashtag)

    return re.sub(r'#\w+', expand, text)


def clean_text(text):
    # Remove retweets and Twitter handles
    text = re.sub(r'RT\s+@\w+:', '', text)  # Remove "RT @user:"
    text = re.sub(r'@\w+', '', text)        # Remove "@user"

    # Remove specific hashtags
    text = re.sub(r'(#NewYearsResolution)', '', text, flags=re.IGNORECASE)

    # Expand hashtags
    text = split_hashtags(text)

    # Remove special characters
    text = re.sub(r'[|*^@~%=+<>×"_/:]', ' ', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def tokenize_sentence(sentence, tokenizer):
    doc = nlp(clean_text(sentence))
    pattern = r'.*\d.*' # filter out pattern

    lemmatized_tokens = [str(token.lemma_).lower() for token in doc if not re.match(pattern, token.lemma_)]
    join_tokens = ' '.join(lemmatized_tokens)

    # Tokenize using BERT tokenizer
    token_list = tokenizer.tokenize(join_tokens)
    combined_sentence = ""

    for token in token_list:
        if len(token) > 1:
            if combined_sentence and not token.startswith("##"): # Check if the token is a subword (starts without a space)
                combined_sentence += " "  # Add a space before new words
            combined_sentence += token.replace("##", "").replace("_", "") # Remove subword marker if present and append the token

    if str(combined_sentence) == '':
        combined_sentence = 'none'

    return lemmatized_tokens


def process_and_save(data_type, input_file, tokenized_file, max_rows=None):
    '''if os.path.isfile(tokenized_file):
        print("tokenized_file already exist!")
    else:'''
    df = pd.read_csv(input_file, sep='\t', encoding='utf-8')

    if max_rows:
        df = df.head(max_rows)

    df["text"] = df["text"].progress_apply(
        lambda x: ' '.join(tokenize_sentence(str(x), tokenizer))
    )
    df.to_csv(tokenized_file, sep='\t', encoding='utf-8', index=False)

    print(f"{data_type.capitalize()} tokenization and lemmatization completed.")
    return


if __name__ == "__main__":
    # Process Train Data
    process_and_save(
        data_type="train",
        input_file="train.csv",
        tokenized_file="tokenized_train.csv",
        max_rows=None # for debug
    )

    # Process Test Data (continuing text_id)
    process_and_save(
        data_type="test",
        input_file="test.csv",
        tokenized_file="tokenized_test.csv",
        max_rows=None # for debug
    )
    print("Tokenization and lemmatization completed successfully.")

    
