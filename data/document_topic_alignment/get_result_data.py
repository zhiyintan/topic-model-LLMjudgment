import ast
import numpy as np
import os
import pandas as pd
import random
import time

def get_file_name_list(path):
    return [file_name for file_name in os.listdir(path)]

def randomly_choose_file(file_list):
    return random.choice(file_list)

def get_files_by_numberoftopic(path, topic_model, number_of_topic_list):
    file_dic = {}
    for number_of_topic in number_of_topic_list:
        pattern = topic_model + '_' + str(number_of_topic)
        file_list = get_file_name_list(path)
        for file in file_list:
            if pattern in file:
                if number_of_topic not in file_dic:
                    file_dic[number_of_topic] = []
                    file_dic[number_of_topic].append(file)
    return file_dic

dataset_list = ['20ng', 'agris', 'tweets_ny']
topic_model_list = ['lda', 'prodlda', 'combinedtm', 'bertopic']
number_of_topic_list = [50, 100]

for dataset in dataset_list: # 'agris', '20ng', 'tweets_ny'
    data_path = f"../raw/{dataset}/"
    train_set_file_path = data_path + 'tokenized_train.csv'

    dataset_result_path = f"../topic_model_output/{dataset}/"
    
    current_date = time.strftime("%Y%m%d%H%M")
    output_file_path = f"sample_100_data_{dataset}_{current_date}.csv"

    # Randomly select an iteration from the topic words results of each topic modeling setup
    def random_select_files():
        file_name_dic = {}
        for topic_model in topic_model_list:
            topic_words_path = dataset_result_path + topic_model + '/topic_words'

            file_dic = get_files_by_numberoftopic(topic_words_path, topic_model, number_of_topic_list)
            for number_of_topic in file_dic:
                selected_filename = randomly_choose_file(file_dic[number_of_topic])
                if number_of_topic not in file_name_dic:
                    file_name_dic[number_of_topic] = []
                file_name_dic[number_of_topic].append(selected_filename)
        return file_name_dic


    # These are the randomly selected topic words result file name for 50 topics and 100 topics
    file_name_dic = random_select_files()
    print(file_name_dic)
    # 20ng: {50: ['lda_50_5_0.5_0.01_100_5000_0_0.5_1_10_1.0_0.csv', 'bertopic_50_10_5_25_0.3_0.csv', 'prodlda_50_200_0.1_0.001_40_0.csv', 'combinedtm_50_200_0.1_0.005_40_0.csv'], 100: ['lda_100_5_0.5_0.01_100_5000_0_0.5_1_10_1.0_0.csv', 'bertopic_100_10_5_13_0.3_0.csv', 'prodlda_100_200_0.1_0.005_40_0.csv', 'combinedtm_100_200_0.1_0.005_40_0.csv']}
    # agris: {50: ['lda_50_5_0.5_0.5_100_5000_0_0.5_1_10_1.0_0.csv', 'bertopic_50_10_5_690_0.3_0.csv', 'prodlda_50_200_0.1_0.005_40_0.csv', 'combinedtm_50_200_0.1_0.005_40_0.csv'], 100: ['lda_100_5_0.5_0.5_100_5000_0_0.5_1_10_1.0_0.csv', 'bertopic_100_10_5_370_0.3_0.csv', 'prodlda_100_200_0.1_0.005_40_0.csv', 'combinedtm_100_200_0.1_0.002_40_0.csv']}
    # tweets_ny: {50: ['lda_50_5_0.5_0.1_100_5000_0_0.5_1_10_1.0_0.csv', 'bertopic_50_25_20_20_0.3_0.csv', 'prodlda_50_200_0.1_0.005_40_0.csv', 'combinedtm_50_200_0.1_0.005_40_0.csv'], 100: ['lda_100_5_0.01_0.01_100_5000_0_0.5_1_10_1.0_0.csv', 'bertopic_100_35_25_9_0.3_0.csv', 'prodlda_100_200_0.5_0.005_40_0.csv', 'combinedtm_100_200_0.3_0.005_40_0.csv']}

    
    # load train set, label
    df_train_set = pd.read_csv(train_set_file_path, sep='\t', encoding='utf-8')
    number_of_train_set = len(df_train_set)
    
    
    # load label file
    label_mapping_path = data_path + 'data_label_mapping.csv'
    labelid_path = data_path + 'label.csv'

    # load data id and label mapping
    textid_labelid_mapping = pd.read_csv(label_mapping_path, sep='\t', encoding='utf-8')
    if dataset == 'agris':
        textid_labelid_mapping['label_id_list'] = textid_labelid_mapping['label_id_list'].apply(ast.literal_eval)
        paperid_to_labelid_dic = dict(zip(textid_labelid_mapping['paper_id'], textid_labelid_mapping['label_id_list']))
    elif dataset == '20ng' or dataset == 'tweets_ny':
        textid_to_labelid_dic = dict(zip(textid_labelid_mapping['text_id'], textid_labelid_mapping['label_id']))

    # load label id and label mapping
    labelid_label_mapping = pd.read_csv(labelid_path, sep='\t', encoding='utf-8')
    labelid_to_label_dic = dict(zip(labelid_label_mapping['label_id'], labelid_label_mapping['label']))

 
    
    rows = []; less_100 = 0
    for number_of_topic in file_name_dic:
        filename_list = file_name_dic[number_of_topic]
        for filename in filename_list:
            topic_model = filename.split('_')[0]
            parameters = filename.rsplit('.', 1)[0]
            

            topic_words_file_path = dataset_result_path + topic_model + '/topic_words/' + filename
            print(topic_words_file_path)
            topic_distribution_file_path = dataset_result_path + topic_model + '/topic_distribution_train/' + parameters + '.npy'
            print(topic_distribution_file_path)

            # Get topic ID to topic words dictionary
            df_topic_words = pd.read_csv(topic_words_file_path, sep='\t', encoding='utf-8')
            topicid_2_topic_dic = dict(zip(df_topic_words['ID'], df_topic_words['Topic words']))

            # Get the highest probability topic ID for each data entry
            dataid_2_topicid_dic = {}
            topic_distribution = np.load(topic_distribution_file_path)
            assigned_topics = np.argmax(topic_distribution, axis=1)
            for i in range(len(assigned_topics)):
                dataid_2_topicid_dic[i] = assigned_topics[i]

            # Group data IDs by topic
            topicid_to_dataids = {}
            for dataid, topicid in enumerate(assigned_topics):
                if topicid not in topicid_to_dataids:
                    topicid_to_dataids[topicid] = []
                topicid_to_dataids[topicid].append(dataid)
            # Sort the dictionary by topicid
            topicid_to_dataids = dict(sorted(topicid_to_dataids.items()))

            # Randomly select 100 data IDs per topic
            for topicid, dataids in topicid_to_dataids.items():
                selected_dataids = random.sample(dataids, min(5, len(dataids)))  # Ensure no more than available data
                # less than 100: prodlda_100: topic 28(92), topic 55(17), topic 73(4), topic 81(74), topic 84(64)

                for dataid in selected_dataids:
                    topicid = dataid_2_topicid_dic[dataid]
                    
                    if dataset == 'agris':
                        paperid = df_train_set.loc[dataid, 'paper_id']
                        labelid_list = [labelid_to_label_dic[labelid].strip() for labelid in paperid_to_labelid_dic[paperid]]
                        row = {
                            'model': topic_model,
                            'number of topics': number_of_topic, 
                            'topic id': topicid,
                            'topic words': topicid_2_topic_dic[topicid],
                            'text id': str(dataid),
                            'paper id': str(paperid),
                            'text': df_train_set.loc[dataid, 'text'],
                            'labels': ' | '.join(labelid_list)
                        }
                    elif dataset == '20ng':
                        textid = df_train_set.loc[dataid, 'text_id']
                        label = [labelid_to_label_dic[textid_to_labelid_dic[textid]].strip()]
                        row = {
                            'model': topic_model,
                            'number of topics': number_of_topic, 
                            'topic id': topicid,
                            'topic words': topicid_2_topic_dic[topicid],
                            'text id': textid,
                            'paper id': '-',
                            'text': df_train_set.loc[dataid, 'text'],
                            'labels': label
                        }
                    elif dataset == 'tweets_ny':
                        textid = df_train_set.loc[dataid, 'text_id']
                        label = [labelid_to_label_dic[textid_to_labelid_dic[dataid]].strip()]
                        row = {
                            'model': topic_model,
                            'number of topics': number_of_topic, 
                            'topic id': topicid,
                            'topic words': topicid_2_topic_dic[topicid],
                            'text id': textid,
                            'paper id': '-',
                            'text': df_train_set.loc[dataid, 'text'],
                            'labels': label
                        }

                    if len(dataids) < 100:
                        less_100 += 1
                        #print(model, number_of_topic, topicid)
                    #print(f"{topicid}\n{topicid_2_topic_dic[topicid]}\n\{dataid}\t{df_train_set.loc[dataid, 'id']}\t{df_train_set.loc[dataid, 'text']}")
                    rows.append(row)
    print(less_100)
    df_sample_data = pd.DataFrame(rows)

    # Save the DataFrame as a tab-separated file
    df_sample_data.to_csv(output_file_path, sep='\t', index=False, encoding='utf-8')

    print(f"Data saved to {output_file_path}")