import json
from txtai.embeddings import Embeddings
import os

embeddings = Embeddings(path='intfloat/e5-base')
embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def information_retrieval(query):
    hits = embeddings.search(query, 5)
    paragraphs = []
    for hit in hits:
        paragraphs.append(hit['text'])
    return paragraphs

def get_wiki_pages(data_path, output_path):
    output_option = 'sparse_retrieval'
    data_file = open(data_path, 'r')
    data = json.load(data_file)
    for idx, case in enumerate(data):
        print(f'Processing {idx+1}/{len(data)}')
        question = case['question']
        case[output_option] = []
        paragraphs = information_retrieval(question)
        case[output_option].append(paragraphs)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def retriever(args):
    get_wiki_pages('./dataset/' + args.task_dataset + '/train.json', './dataset/' + args.task_dataset + '/train_wiki_retriever.json')
    get_wiki_pages('./dataset/' + args.task_dataset + '/test.json', './dataset/' + args.task_dataset + '/test_wiki_retriever.json')