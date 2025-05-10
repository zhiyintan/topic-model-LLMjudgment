import pandas as pd
import random
import nltk
from nltk.corpus import wordnet

# Download NLTK data (ensure this runs once before using it)
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_word_type(word):
    """Function to identify if a word is a noun or adjective."""
    word_tag = nltk.pos_tag([word])[0][1]
    if word_tag.startswith('NN'):  # Noun
        return 'noun'
    elif word_tag.startswith('JJ'):  # Adjective
        return 'adjective'
    else:
        return None

def select_anchor(topic_words):
    """Select an anchor word based on the rules."""
    nouns = []
    adjectives = []
    for word in topic_words:
        word_type = get_word_type(word)
        if word_type == 'noun':
            nouns.append(word)
        elif word_type == 'adjective':
            adjectives.append(word)
    
    # Choose a noun if available, otherwise fallback to an adjective
    if nouns:
        return random.choice(nouns)
    elif adjectives:
        return random.choice(adjectives)
    else:
        return random.choice(topic_words)  # Fallback: Any random word
    


# Load the datasets
#fname_20ng = "random_100_topic_words_20ng.csv"
#fname_agris = "random_100_topic_words_agris.csv"
fname_tweetsnyr = "random_100_topic_words_tweetsnyr.csv"

#df_20ng = pd.read_csv(fname_20ng, sep="\t", encoding='utf-8')
#df_agris = pd.read_csv(fname_agris, sep="\t", encoding='utf-8')
df_tweetsnyr = pd.read_csv(fname_tweetsnyr, sep="\t", encoding='utf-8', dtype={"ID": str})


# Process each row in the DataFrame
anchors = []
for i in range(len(df_tweetsnyr)):
    topic_words_list = df_tweetsnyr.loc[i, "Topic words"].split(' ')
    anchor = select_anchor(topic_words_list)
    anchors.append(anchor)

# Add the anchor column
df_tweetsnyr['Anchor'] = anchors

# Save the modified DataFrame
df_tweetsnyr.to_csv("random_100_topic_words_tweetsnyr_with_anchors.csv", sep="\t", index=False)
