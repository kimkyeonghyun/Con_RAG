import os
import sys
import logging
import json
import torch
import argparse
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    T5ForSequenceClassification,
    T5Config,
)
from utils.utils import TqdmLoggingHandler, write_log, get_huggingface_model_name, get_wandb_exp_name, get_torch_device
from utils.metric import f1_cal, gold_answer, answer_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_num_threads(2)  


def create_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(model_name, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        max_memory={"0": "10000MB"},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  
    return model, tokenizer


def classifier_model(args: argparse.Namespace):
    device = get_torch_device(args.device)

    logger = logging.getLogger(__name__)
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)

    write_log(logger, 'Building model')
    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    config.num_labels = 1
    classifier = T5ForSequenceClassification.from_pretrained(model_name, config=config)

    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type, 'final_model.pt')
    write_log(logger, "Loading model weights")
    checkpoint = torch.load(load_model_name, map_location=torch.device('cpu'))

    for key in list(checkpoint['model'].keys()):
        if 'model.' in key:
            checkpoint['model'][key.replace('model.', '')] = checkpoint['model'].pop(key)

    classifier.load_state_dict(checkpoint['model'])
    classifier = classifier.to(device)
    write_log(logger, f'Loaded model weights from {load_model_name}')
    return classifier, tokenizer


def generate_answer(model, tokenizer, question, doc=None, is_retrieved=False):
    if is_retrieved:
        inputs = f"Please refer to the given document and answer the question: \n Question: {question} \nRelate passage: {doc} \nAnswer: "
    else:
        inputs = f"Please answer the question: \n Question: {question} \nAnswer: "
    
    inputs = tokenizer(inputs, return_tensors='pt').to(device)
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0])
    return answer.split('Answer:')[1] if 'Answer:' in answer else 'No answer found'


def evaluate_model(args, data, model, tokenizer, classifier, classifier_tokenizer):
    pred_ans = []
    for question in tqdm(data):
        my_question = question["question"]
        ans = question["answer"]

        # Classify the question using the classifier
        inputs = classifier_tokenizer(my_question, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = classifier(**inputs)
        logits = outputs.logits.squeeze().detach().cpu().item()
        is_retrieved = logits > 0.5  # Set threshold for binary classification (1 = retrieve, 0 = no retrieve)

        if is_retrieved:
            document = ''.join(question["sparse_retrieval"][0])
        else:
            document = None  # No retrieval if not relevant

        predicted_answer = generate_answer(model, tokenizer, my_question, doc=document, is_retrieved=is_retrieved)

        pred_ans.append([ans, predicted_answer])
        question['pred_answer'] = [predicted_answer]
        question['accuracy'] = [answer_accuracy(predicted_answer, ans)]

    return pred_ans


def classifier_llm_generation(args):
    dir_path = f'./dataset/{args.task_dataset}'
    data_path = os.path.join(dir_path, 'test_wiki_retriever.json')

    with open(data_path, 'r') as data_file:
        data = json.load(data_file)

    # Load LLM model and tokenizer
    model_name = get_huggingface_model_name(args.llm_model)
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)
    model.eval()

    # Load classifier model and tokenizer
    classifier, classifier_tokenizer = classifier_model(args)

    # Evaluate the model
    pred_ans = evaluate_model(args, data, model, tokenizer, classifier, classifier_tokenizer)

    # Calculate F1 score and accuracy
    f1 = f1_cal(args, pred_ans)
    all_accuracy = gold_answer(args, pred_ans)
    print(f"F1 Score: {f1}, All Accuracy: {all_accuracy}")

    # Save results
    data[0]['f1_score'] = [f1]
    data[0]['all_accuracy'] = [all_accuracy]

    result_path = os.path.join(dir_path, f'test_results_{args.model_type}.json')
    with open(result_path, 'w') as result_file:
        json.dump(data, result_file, indent=4)