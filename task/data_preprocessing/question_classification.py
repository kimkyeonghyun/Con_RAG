import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from captum.attr import ShapleyValues, LLMAttribution, TextTemplateInput
from utils.metric import f1_cal, gold_answer, answer_accuracy
from utils.utils import get_huggingface_model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, bnb_config):
    max_memory = "10000MB"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: max_memory},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Needed for LLaMA tokenizer
    return model, tokenizer


def create_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def generate_answer(model, tokenizer, prompt, max_length=4096):
    inputs = tokenizer(prompt, max_length=max_length, return_tensors='pt').to(device)
    outputs = model.generate(**inputs)
    decoded_output = tokenizer.decode(outputs[0])
    return decoded_output.split('Answer:')[1] if 'Answer:' in decoded_output else 'No answer found'


def evaluate_and_classify(question, template, model, tokenizer, no_doc_list, ret_doc_list):
    my_question = question["question"]
    doc = ''.join(question["sparse_retrieval"][0])
    ans = question['answer']

    prompt = template.format(my_question, doc)
    predicted_answer = generate_answer(model, tokenizer, prompt)

    accuracy = answer_accuracy(predicted_answer, ans)
    if accuracy > 0:
        sv = ShapleyValues(model)
        llm_attr = LLMAttribution(sv, tokenizer)
        inp = TextTemplateInput(template=template, values=[my_question, doc])
        attr_res = llm_attr.attribute(inp, target=predicted_answer)
        if attr_res.seq_attr[0] > attr_res.seq_attr[1]:
            no_doc_list.append(my_question)
        else:
            ret_doc_list.append(my_question)

    question['pred_answer'] = predicted_answer
    question['accuracy'] = accuracy
    return predicted_answer


def question_classification(args):
    # Main execution
    model_name = get_huggingface_model_name(args.llm_model)
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)
    model.eval()
    dir_path = f'./dataset/{args.task_dataset}'
    data_path = f'{dir_path}/train_wiki_retriever.json'
    with open(data_path, 'r') as data_file:
        data = json.load(data_file)

    pred_ans, no_doc_list, ret_doc_list = [], [], []

    template = 'Please refer to the given document and answer the question: \n Question: {}? \nRelate passage: {} \nAnswer:'

    for question in tqdm(data):
        predicted_answer = evaluate_and_classify(
            question, template, model, tokenizer, no_doc_list, ret_doc_list
        )
        pred_ans.append([question['answer'], predicted_answer])

    # Metrics calculation
    f1 = f1_cal(args, pred_ans)
    exact_match = gold_answer(args, pred_ans)
    print(f"F1 Score: {f1}, Exact Match: {exact_match}")

    data[0]['f1_score'], data[0]['exat'] = [f1], [exact_match]

    # Update classification results
    question_class_path = f'{dir_path}/train_question_classification.json'
    with open(question_class_path, 'r') as question_file:
        question_list = json.load(question_file)

    question_list['no_doc'] += no_doc_list
    question_list['ret_doc'] += ret_doc_list

    # Save results
    with open(f'{dir_path}/retrieval_train_results_{args.model_type}-7B.json', 'w') as f:
        json.dump(data, f, indent=4)
    with open(question_class_path, 'w') as f:
        json.dump(question_list, f, indent=4)


