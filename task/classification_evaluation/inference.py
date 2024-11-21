import os
import torch
from txtai.embeddings import Embeddings
from transformers import AutoTokenizer, T5ForSequenceClassification, T5Config, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Embeddings 설정
embeddings = Embeddings(path='intfloat/e5-base')
embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")


def create_classifier(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    config.num_labels = 1
    classifier = T5ForSequenceClassification.from_pretrained(model_name, config=config)
    classifier.to(device)
    classifier.eval()
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
    classifier_name = args.classifier_model
    classifier, classifier_tokenizer = create_classifier(classifier_name)
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