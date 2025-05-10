import pandas as pd

dataset_list = ['20ng', 'agris', 'tweets_ny']

for dataset in dataset_list:
    data_path = f"./{dataset}/"
    train_set_file_path = data_path + 'tokenized_train.csv'
    test_set_file_path = data_path + 'tokenized_test.csv'
    label_set_file_path = data_path + 'label.csv'
    
    all_token_count = 0
    train_df = pd.read_csv(train_set_file_path, sep='\t', encoding='utf-8')
    test_df = pd.read_csv(test_set_file_path, sep='\t', encoding='utf-8')

    for text in train_df['text']:
        all_token_count += len(text.split())
        
    for text in test_df['text']:
        all_token_count += len(text.split())

    average_token_count = all_token_count / (len(train_df) + len(test_df))
    print(f"Dataset: {dataset}")
    print(f"All token count: {all_token_count}")
    print(f"Average token count: {average_token_count}")


    label_df = pd.read_csv(label_set_file_path, sep='\t', encoding='utf-8')
    label_count = len(label_df)
    print(f"Label count: {label_count}")
    




