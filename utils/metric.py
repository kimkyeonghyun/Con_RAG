from rouge import Rouge
import string
from collections import Counter

import re

def rouge_cal(args, pred_ans):
    rouge_score = 0
    rouge = Rouge()
    for i in range(len(pred_ans)):
        rouge_score += rouge.get_scores(pred_ans[i][1], pred_ans[i][0], avg=True)
    return rouge_score/len(pred_ans)

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def rouge_cal(args, pred_ans):
    rouge_score = 0
    rouge = Rouge()
    for i in range(len(pred_ans)):
        prediction_tokens = normalize_answer(pred_ans[i][1])
        ground_truth_tokens = normalize_answer(pred_ans[i][0])
        rouge_score += rouge.get_scores(prediction_tokens, ground_truth_tokens, avg=True)
    return rouge_score/len(pred_ans)


def f1_cal(args, pred_ans):
    f1 = 0
    for i in range(len(pred_ans)):
        f1 += f1_score(pred_ans[i][1], pred_ans[i][0])
    return f1/len(pred_ans)

def gold_answer(args, pred_ans):
    score = 0
    for i in range(len(pred_ans)):
        prediction_tokens = set(normalize_answer(pred_ans[i][1]).split())
        ground_truth_tokens = set(normalize_answer(pred_ans[i][0]).split())
        
        if not ground_truth_tokens:
            match_score = 0

        # Check if all ground truth tokens are in the prediction tokens
        if ground_truth_tokens.issubset(prediction_tokens):
            match_score = 1
        else:
            match_score = 0
        score+=match_score
    return score/len(pred_ans)

def answer_accuracy(prediction, ground_truth):

    prediction_tokens = set(normalize_answer(prediction).split())
    ground_truth_tokens = set(normalize_answer(ground_truth).split())
    
    if not ground_truth_tokens:
        return 0

    # Check if all ground truth tokens are in the prediction tokens
    if ground_truth_tokens.issubset(prediction_tokens):
        return 1
    else:
        return 0
        
