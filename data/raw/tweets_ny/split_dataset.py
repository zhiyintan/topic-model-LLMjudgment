import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import Counter

def clean_text(text):
    if pd.isnull(text):
        return ""
    # Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove newline, tab, carriage returns
    text = re.sub(r'[\n\r\t]+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load the dataset
df = pd.read_csv("new_year_resolutions.csv", sep=';', encoding='utf-8')
df['text'] = df['text'].apply(clean_text)

# Split the labels into a list
df['labels'] = df['resolution_category'] #df['resolution_topics']
print(f"{len(df)}:\t Original size of data set")

# Count labels
label_counts = Counter([label for label in df['labels']])
print(f"{len(label_counts)}:\t Counts of resolution_topics")

# Generate unique IDs for each data entry
df['text_id'] = range(1, len(df) + 1)

# Generate unique IDs for each label
unique_labels = set(label for label in df['labels'])
label_to_id = {label: idx for idx, label in enumerate(unique_labels, start=1)}

# Create a label mapping DataFrame
label_df = pd.DataFrame(list(label_to_id.items()), columns=['label', 'label_id'])
label_df = label_df[['label_id', 'label']]  # Ensure column order is label_id, label

# Map data IDs to label ID lists
df['label_id'] = df['labels'].apply(lambda x: label_to_id[x])

# Create the mapping DataFrame with text_id and label_id_list
mapping_df = df[['text_id', 'label_id']]

# Remove unnecessary columns for the data file
data_df = df[['text_id', 'text']]

# Split the data into train and test sets
train_set, test_set = train_test_split(data_df, test_size=0.1, random_state=42)

# Output information
print(f"{len(train_set)}:\t Final train set size")
print(f"{len(test_set)}:\t Final test set size")

# Save outputs to files
#data_df.to_csv("data.csv", sep='\t', encoding='utf-8', index=False)
label_df.to_csv("label.csv", sep='\t', encoding='utf-8', index=False)
mapping_df.to_csv("data_label_mapping.csv", sep='\t', encoding='utf-8', index=False)
train_set.to_csv("train.csv", sep='\t', encoding='utf-8', index=False)
test_set.to_csv("test.csv", sep='\t', encoding='utf-8', index=False)
