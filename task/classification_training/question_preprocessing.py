# text_classification
# preprocessing

# 라이브러리
import os
import sys
import pickle
import argparse
import json
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_json(file_path):
    data = []
    with open(file_path, 'r') as file:
        data.append(json.load(file))
    return data

def load_data(args: argparse.Namespace):

    name = args.task_dataset.lower()
    train_valid_split = args.train_valid_split

    train_data = {
        'question': [],
        'labels': []
    }
    valid_data = {
        'question': [],
        'labels': []
    }
    test_data = {
        'question': [],
        'labels': []
    }

    if name == 'triviaqa':
        dataset = load_json('./dataset/' + name + '/train_question_classification.json')
        df_no_doc = pd.DataFrame(dataset[0]['no_doc'], columns=['question'])
        df_no_doc['labels'] = 0

        df_ret_doc = pd.DataFrame(dataset[0]['ret_doc'], columns=['question'])
        df_ret_doc['labels'] = 1

        ret_valid_cutoff = int(len(df_ret_doc) * (1-train_valid_split))
        ret_test_cutoff = int(len(df_ret_doc) * train_valid_split)
        ret_train_valid_df = df_ret_doc[:ret_valid_cutoff]
        ret_train_df = ret_train_valid_df[ret_test_cutoff:]
        ret_valid_df = ret_train_valid_df[:ret_test_cutoff]
        ret_test_df = df_ret_doc[ret_valid_cutoff:]

        no_valid_cutoff = int(len(df_no_doc) * (1-train_valid_split))
        no_test_cutoff = int(len(df_no_doc) * train_valid_split)
        no_train_valid_df = df_no_doc[:no_valid_cutoff]
        no_train_df = no_train_valid_df[no_test_cutoff:]
        no_valid_df = no_train_valid_df[:no_test_cutoff]
        no_test_df = df_no_doc[no_valid_cutoff:]
        train_df = pd.concat([ret_train_df, no_train_df], axis = 0)
        valid_df = pd.concat([ret_valid_df, no_valid_df], axis = 0)
        test_df = pd.concat([ret_test_df, no_test_df], axis=0)

        train_df = train_df.sample(frac = 1).reset_index(drop = True) 
        valid_df = valid_df.sample(frac = 1).reset_index(drop = True)
        test_df = test_df.sample(frac = 1).reset_index(drop = True)

        train_data['question'] = train_df['question'].tolist()
        train_data['labels'] = train_df['labels'].tolist()
        valid_data['question'] = valid_df['question'].tolist()
        valid_data['labels'] = valid_df['labels'].tolist()
        test_data['question'] = test_df['question'].tolist()
        test_data['labels'] = test_df['labels'].tolist()

    elif name == 'naturalqa':
        dataset = load_json('./dataset/' + name + '/train_question_classification.json')
        df_no_doc = pd.DataFrame(dataset[0]['no_doc'], columns=['question'])
        df_no_doc['labels'] = 0

        df_ret_doc = pd.DataFrame(dataset[0]['ret_doc'], columns=['question'])
        df_ret_doc['labels'] = 1

        num_classes = 2

        ret_valid_cutoff = int(len(df_ret_doc) * (1-train_valid_split))
        ret_test_cutoff = int(len(df_ret_doc) * train_valid_split)
        ret_train_valid_df = df_ret_doc[:ret_valid_cutoff]
        ret_train_df = ret_train_valid_df[ret_test_cutoff:]
        ret_valid_df = ret_train_valid_df[:ret_test_cutoff]
        ret_test_df = df_ret_doc[ret_valid_cutoff:]

        no_valid_cutoff = int(len(df_no_doc) * (1-train_valid_split))
        no_test_cutoff = int(len(df_no_doc) * train_valid_split)
        no_train_valid_df = df_no_doc[:no_valid_cutoff]
        no_train_df = no_train_valid_df[no_test_cutoff:]
        no_valid_df = no_train_valid_df[:no_test_cutoff]
        no_test_df = df_no_doc[no_valid_cutoff:]
        train_df = pd.concat([ret_train_df, no_train_df], axis = 0)
        valid_df = pd.concat([ret_valid_df, no_valid_df], axis = 0)
        test_df = pd.concat([ret_test_df, no_test_df], axis=0)

        train_df = pd.concat([ret_train_df, no_train_df], axis = 0)
        valid_df = pd.concat([ret_valid_df, no_valid_df], axis = 0)
        test_df = pd.concat([ret_test_df, no_test_df], axis=0)

        train_df = train_df.sample(frac = 1).reset_index(drop = True) 
        valid_df = valid_df.sample(frac = 1).reset_index(drop = True)
        test_df = test_df.sample(frac = 1).reset_index(drop = True)

        train_data['question'] = train_df['question'].tolist()
        train_data['labels'] = train_df['labels'].tolist()
        valid_data['question'] = valid_df['question'].tolist()
        valid_data['labels'] = valid_df['labels'].tolist()
        test_data['question'] = test_df['question'].tolist()
        test_data['labels'] = test_df['labels'].tolist()

    elif name == 'squad':
        dataset = load_json('./dataset/' + name + '/train_question_classification.json')
        df_no_doc = pd.DataFrame(dataset[0]['no_doc'], columns=['question'])
        df_no_doc['labels'] = 0

        df_ret_doc = pd.DataFrame(dataset[0]['ret_doc'], columns=['question'])
        df_ret_doc['labels'] = 1

        num_classes = 2
       
        ret_valid_cutoff = int(len(df_ret_doc) * (1-train_valid_split))
        ret_test_cutoff = int(len(df_ret_doc) * train_valid_split)
        ret_train_valid_df = df_ret_doc[:ret_valid_cutoff]
        ret_train_df = ret_train_valid_df[ret_test_cutoff:]
        ret_valid_df = ret_train_valid_df[:ret_test_cutoff]
        ret_test_df = df_ret_doc[ret_valid_cutoff:]

        no_valid_cutoff = int(len(df_no_doc) * (1-train_valid_split))
        no_test_cutoff = int(len(df_no_doc) * train_valid_split)
        no_train_valid_df = df_no_doc[:no_valid_cutoff]
        no_train_df = no_train_valid_df[no_test_cutoff:]
        no_valid_df = no_train_valid_df[:no_test_cutoff]
        no_test_df = df_no_doc[no_valid_cutoff:]
        train_df = pd.concat([ret_train_df, no_train_df], axis = 0)
        valid_df = pd.concat([ret_valid_df, no_valid_df], axis = 0)
        test_df = pd.concat([ret_test_df, no_test_df], axis=0)

        train_df = pd.concat([ret_train_df, no_train_df], axis = 0)
        valid_df = pd.concat([ret_valid_df, no_valid_df], axis = 0)
        test_df = pd.concat([ret_test_df, no_test_df], axis=0)

        train_df = train_df.sample(frac = 1).reset_index(drop = True) 
        valid_df = valid_df.sample(frac = 1).reset_index(drop = True)
        test_df = test_df.sample(frac = 1).reset_index(drop = True)

        train_data['question'] = train_df['question'].tolist()
        train_data['labels'] = train_df['labels'].tolist()
        valid_data['question'] = valid_df['question'].tolist()
        valid_data['labels'] = valid_df['labels'].tolist()
        test_data['question'] = test_df['question'].tolist()
        test_data['labels'] = test_df['labels'].tolist()
    return train_data, valid_data, test_data, num_classes

def preprocessing(args: argparse.Namespace) -> None:

    # Load data
    train_data, valid_data, test_data, num_classes = load_data(args)

    model = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model)

    data_dict = {
        'train': {
            'question': [],
            'labels': [],
            'num_classes': num_classes,
        },
        'valid': {
            'question': [],
            'labels': [],
            'num_classes': num_classes,
        },
        'test': {
            'question': [],
            'labels': [],
            'num_classes': num_classes,
        }
    }

    preprocessed_path = './dataset/' + args.task_dataset + '/classification/'
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in tqdm(range(len(split_data['question'])), desc= 'Preprocessing', position=0, leave=True):
            text = split_data['question'][idx]
            label = split_data['labels'][idx]

            token = tokenizer(text, padding='max_length', truncation=True,
                               max_length=args.max_seq_len, return_tensors='pt')
            data_dict[split]['question'].append(token['input_ids'].squeeze())
            data_dict[split]['labels'].append(torch.tensor(label, dtype=torch.long))
            


        with open(os.path.join(preprocessed_path, args.model_type, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)