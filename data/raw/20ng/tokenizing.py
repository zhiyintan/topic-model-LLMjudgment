import os
import re
import string
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import spacy

# Initialize tqdm progress bar
tqdm.pandas()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # Remove special char
    text = re.sub(r'[|*^#@~%=+<>]{1,}', ' ', text)
    return text


def tokenize_sentence(sentence, tokenizer):
    """
    Clean, tokenize, and lemmatize a sentence.
    Applies multiple filters and returns a space-separated string of cleaned tokens.
    """

    # Lemmatize using SpaCy
    doc = nlp(clean_text(sentence))

    pattern = r'.*\d.*' # filter out pattern
    lemmatized_tokens = [token.lemma_ for token in doc if not re.match(pattern, token.lemma_)]
    join_tokens = ' '.join(lemmatized_tokens)
    
    # Tokenize using BERT tokenizer
    token_list = tokenizer.tokenize(join_tokens)

    combined_sentence = ""
    for token in token_list:
        if len(token) > 1:
            if combined_sentence and not token.startswith("##"):  # New word, add space
                combined_sentence += " "
            combined_sentence += token.replace("##", "").replace("_", "")  # Remove subword markers

    if str(combined_sentence) == '':
        combined_sentence = 'none'
    return combined_sentence


def tokenize_dataframe(df, output_filename):
    """
    Apply tokenization and lemmatization to an entire DataFrame.
    Saves the processed DataFrame as a CSV.
    """
    print(f"Processing {output_filename}...")
    
    # Ensure 'text' column exists
    if "text" not in df.columns:
        raise ValueError(f"Error: Column 'text' not found in DataFrame!")

    # Apply tokenization with progress tracking
    df["text"] = df["text"].progress_apply(lambda x: tokenize_sentence(str(x), tokenizer))

    # Save the tokenized dataset
    df.to_csv(output_filename, sep='\t', index=False, encoding="utf-8")
    print(f"✅ {output_filename} saved successfully!")

if __name__ == "__main__":
    # Load train and test datasets
    train_file = "train.csv"
    test_file = "test.csv"
    tokenized_train_file = "tokenized_train.csv"
    tokenized_test_file = "tokenized_test.csv"

    if os.path.isfile(train_file):
        df_train = pd.read_csv(train_file, sep='\t', encoding="utf-8")
        tokenize_dataframe(df_train, tokenized_train_file)

    if os.path.isfile(test_file):
        df_test = pd.read_csv(test_file, sep='\t', encoding="utf-8")
        tokenize_dataframe(df_test, tokenized_test_file)

    print("🎯 Tokenization and lemmatization completed successfully!")

    #sentence = "Huh? ^^^^^ Watch it, where k = P(x | A) / P(x) = P(x | B) / P(x) _T_h_i_s _p_r_e_s_e_ ////// misc. /////// the wedding planner 49 / 10 SOFTWARE YARD SALE 1993/04/28 17:47:38 284,292 **** --- 284,298"
    #print(tokenize_sentence(sentence, tokenizer))
