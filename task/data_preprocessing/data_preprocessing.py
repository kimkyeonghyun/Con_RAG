import os
import sys
import argparse
import jsonlines
import json
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_jsonlines(file_path):
    data = []
    # Open the file using jsonlines.open directly with the path
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def load_json(file_path):
    data = []
    # Open the file using jsonlines.open directly with the path
    with open(file_path, 'r') as file:
        data.append(json.load(file))
    return data

def load_data(args: argparse.Namespace):
    name = args.task_dataset.lower()

    if name == 'commonsense':
        dir_path = './dataset/' + name
        train_path = dir_path + '/train_rand_split.jsonl'
        test_path = dir_path + '/dev_rand_split.jsonl'
        train_data = load_jsonlines(train_path)
        test_data = load_jsonlines(test_path)

    if name == 'triviaqa':
        dir_path = './dataset/' + name
        train_path = dir_path + '/qa/wikipedia-train.json'
        test_path = dir_path + '/qa/wikipedia-dev.json'
        train_data = load_json(train_path)
        test_data = load_json(test_path)

    elif name == 'naturalqa':
        dir_path = './dataset/' + name
        train_path = dir_path + '/nq-train.json'
        test_path = dir_path + '/nq-dev.json'
        train_data = load_json(train_path)
        test_data = load_json(test_path)

    elif name == 'squad':
        dir_path = './dataset/' + name
        train_path = dir_path + '/squad1-train.json'
        test_path = dir_path + '/squad1-dev.json'
        train_data = load_json(train_path)
        test_data = load_json(test_path)
    return train_data, test_data

def transform_data(args, data):
    new_data = []
    if args.task_dataset.lower() == 'commonsense':
        for i in data:

            trans_data = {
            'id': i['id'],
            'question': f"{i['question']['stem']}\n",
            'Choice': f" A) {i['question']['choices'][0]['text']}\n" +
                      f" B) {i['question']['choices'][1]['text']}\n" +
                      f" C) {i['question']['choices'][2]['text']}\n" +
                      f" D) {i['question']['choices'][3]['text']}\n" +
                      f" E) {i['question']['choices'][3]['text']}\n",
            'answerKey': i['answerKey']
            }
            new_data.append(trans_data)

    elif args.task_dataset.lower() == 'strategy':
        for i in data:

            trans_data = {
            'id': i['id'],
            'question': f"{i['question']['stem']}\n",
            'answerKey': i['answerKey']
            }
            new_data.append(trans_data)

    elif args.task_dataset.lower() == 'triviaqa':
        trans_table = str.maketrans({
            ':': '_',
            '"': '_',
            '*': '_'
        })
        for i in data[0]['Data']:
            doc_list = []
            for j in range(len(i['EntityPages'])):
                trans_path = i['EntityPages'][j]['Filename'].translate(trans_table)
                doc_list.append('./dataset/triviaqa/evidence/wikipedia/'+trans_path)
            trans_data = {
                'id': i['QuestionId'],
                'question': i['Question'],
                'doc': doc_list,
                'answer': i['Answer']['Value']
            }
            new_data.append(trans_data)

    elif args.task_dataset.lower() == 'naturalqa':
        for i in data[0]:
            trans_data = {
            'question': i['question'],
            'answer': i['answers'][0]
            }
            new_data.append(trans_data)

    elif args.task_dataset.lower() == 'squad':
        for i in data[0]:
            trans_data = {
            'question': i['question'],
            'answer': i['answers'][0]
            }
            new_data.append(trans_data)

    return new_data




def preprocessing(args: argparse.Namespace) -> None:
    # Load data
    train_data, test_data= load_data(args)
    train_data = transform_data(args, train_data)
    test_data = transform_data(args, test_data)
    with open('./dataset/'+ args.task_dataset.lower() + '/train.json', 'w') as json_file:
        json.dump(train_data, json_file, indent=4)
    with open('./dataset/'+ args.task_dataset.lower() + '/test.json', 'w') as json_file:
        json.dump(test_data, json_file, indent=4)
    