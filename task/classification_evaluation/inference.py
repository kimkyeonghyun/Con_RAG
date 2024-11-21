import os
import torch
import logging
from txtai.embeddings import Embeddings
from transformers import AutoTokenizer, T5ForSequenceClassification, T5Config, AutoModelForCausalLM
from utils.utils import TqdmLoggingHandler, write_log, get_huggingface_model_name, get_torch_device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# # Embeddings 설정
# embeddings = Embeddings(path='intfloat/e5-base')
# embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")


def classifier_model(args):
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


def classify_question(classifier, tokenizer, question):
    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True).to(device)
    outputs = classifier(**inputs)
    logits = outputs.logits.squeeze().detach().cpu().item()
    return logits > 0.5  # Relevant 여부


def information_retrieval(query):
    hits = embeddings.search(query, 5)
    return [hit['text'] for hit in hits]


def load_llm(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question, doc=None):
    inputs = f"Please refer to the given document and answer the question:\nQuestion: {question}\nDocument: {doc}\nAnswer:" \
        if doc else f"Please answer the question:\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(inputs, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split('Answer:')[1] if 'Answer:' in answer else 'No answer found'


def inference(args):
    print(f"Starting inference for question: {args.question}")

    # Classifier
    classifier, classifier_tokenizer = classifier_model(args)
    is_relevant = classify_question(classifier, classifier_tokenizer, args.question)

    if not is_relevant:
        print("Classification Result: Not relevant")
        print("Skipping retrieval and proceeding with LLM only.")

        # LLM Inference without retrieval
        llm_name = args.llm_model
        llm_model, llm_tokenizer = load_llm(llm_name)
        answer = generate_answer(llm_model, llm_tokenizer, args.question)

        print(f"Generated Answer: {answer}")
        return

    print("Classification Result: Relevant")

    # Retrieval (is_relevant이 True일 때만 실행)
    retrieved_docs = information_retrieval(args.question)
    combined_docs = " ".join(retrieved_docs) if retrieved_docs else None
    print(f"Retrieved Documents: {retrieved_docs}")

    # LLM Inference
    llm_name = args.llm_model
    llm_model, llm_tokenizer = load_llm(llm_name)
    answer = generate_answer(llm_model, llm_tokenizer, args.question, doc=combined_docs)

    print(f"Generated Answer: {answer}")