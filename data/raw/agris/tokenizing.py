import os
import re
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

import spacy
nlp = spacy.load("en_core_web_sm")

from wtpsplit import SaT
sat = SaT("sat-3l-sm")
#sat.half().to("cuda")


# remove noise from the start of text
def remove_noise(text):
    # Use a set for lowercased unwanted phrases
    unwanted_phrases = {'synopsis', 'introduction', 'abstract'}
    lower_text = text.lower()  # Convert the text to lowercase once
    for phrase in unwanted_phrases:
        if lower_text.startswith(phrase):
            return text[len(phrase):].strip()
    return text

# sentencizer
def split_sentence(text):
    return sat.split(text)
        
def sentenizer(df, start_text_id=0):
    df['abstract'] = df['abstract'].fillna('') # Handle missing values in 'abstract' column
    df['abstract'] = df['abstract'].apply(remove_noise) # Apply the function to the 'abstract' column
    df['abstract_sentences'] = df['abstract'].progress_apply(lambda x: split_sentence(x))# Process abstracts to split into sentences

    # Build the new DataFrame efficiently
    rows = []
    text_id = start_text_id
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        rows.append({'text_id': text_id, 'paper_id': row['paper_id'], 'text': row['title']})  # Add the title with its paper_id
        text_id += 1
        for sentence in row['abstract_sentences']:
            if len(sentence.split(' ')) > 2:  # Filter short sentences
                rows.append({'text_id': text_id, 'paper_id': row['paper_id'], 'text': sentence})
                text_id += 1

    sentenized_df = pd.DataFrame(rows) # Create the new DataFrame
    return sentenized_df, text_id

'''def tokenize_sentence(sentence, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    combined_sentence = ""

    for token in tokens:
        if combined_sentence and not token.startswith("##"): # Check if the token is a subword (starts without a space)
            combined_sentence += " "  # Add a space before new words
        combined_sentence += token.replace("##", "").replace("_", "") # Remove subword marker if present and append the token

    doc = nlp(combined_sentence)
    lemmatized_tokens = [token.lemma_ for token in doc]

    return lemmatized_tokens'''

def clean_text(text):
    # Remove special char
    text = re.sub(r'[|*^#@~%=+<>×]{1,}', ' ', text)
    return text

def tokenize_sentence(sentence, tokenizer):
    doc = nlp(clean_text(sentence))
    pattern = r'.*\d.*' # filter out pattern

    lemmatized_tokens = [token.lemma_ for token in doc if not re.match(pattern, token.lemma_)]
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


def process_and_save(data_type, input_file, sentenized_file, tokenized_file, start_text_id=0, max_rows=None):
    if os.path.isfile(sentenized_file):
        sentenized_df = pd.read_csv(sentenized_file, sep='\t', encoding='utf-8')
        last_text_id = sentenized_df['text_id'].max() + 1
    else:
        df = pd.read_csv(input_file, sep='\t', encoding='utf-8')

        if max_rows:
            df = df.head(max_rows)  # Use only the first `max_rows` for testing

        sentenized_df, last_text_id = sentenizer(df, start_text_id)
        sentenized_df.to_csv(sentenized_file, sep='\t', encoding='utf-8', index=False)

    if not os.path.isfile(tokenized_file):
        sentenized_df["text"] = sentenized_df["text"].progress_apply(
            lambda x: ' '.join(tokenize_sentence(str(x), tokenizer))
        )
        sentenized_df.to_csv(tokenized_file, sep='\t', encoding='utf-8', index=False)

    print(f"{data_type.capitalize()} tokenization and lemmatization completed.")
    return last_text_id


if __name__ == "__main__":
    # Process Train Data
    next_text_id = process_and_save(
        data_type="train",
        input_file="train.csv",
        sentenized_file="sentenized_train.csv",
        tokenized_file="tokenized_train.csv",
        start_text_id=0,
        max_rows=None # for debug
    )

    # Process Test Data (continuing text_id)
    process_and_save(
        data_type="test",
        input_file="test.csv",
        sentenized_file="sentenized_test.csv",
        tokenized_file="tokenized_test.csv",
        start_text_id=next_text_id,
        max_rows=None # for debug
    )
    print("Tokenization and lemmatization completed successfully.")

    
