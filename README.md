# Con-RAG: Contribution-based Analysis of Retrieval Necessity for Efficient Retrieval Augmentation Generation

This repository contains the implementation of **Con-RAG**, a novel approach to optimize retrieval-augmented generation (RAG) by determining the necessity of retrieval using contribution analysis. The method leverages Shapley value to evaluate the contribution of questions and retrieved information, improving computational efficiency and accuracy in response generation.

---

**Con-RAG** addresses the limitations of traditional RAG systems, which often perform unnecessary retrievals, by:
1. **Determining Retrieval Necessity**:
   - Uses Shapley value to calculate the contributions of questions and retrieved data.
   - Classifies whether retrieval is necessary or not.
2. **Improving Efficiency**:
   - Reduces computational overhead by avoiding unnecessary retrievals.
   - Achieves high accuracy with fewer model parameters.

---

# Pipeline Steps

This section describes the steps to execute the Con-RAG pipeline, including preprocessing, retrieval, question classification, training, and evaluation.

---

## 1. Preprocessing, Retrieval, and Question Classification

```bash
python main.py --task=preprocessing --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=preprocessing --job=retriever --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=preprocessing --job=question_classification --task_dataset=${DATASET} --model_type=${MODEL} --llm_model=${LLM}
```

## 2. Train and Test the Classification Model

```bash
python main.py --task=classification_training --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification_training --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}
python main.py --task=classification_training --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}
```
## 3. Generation Testing
```bash
python main.py --task=classification_evaluation --job=generation_test --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE} --llm_model=${LLM}
```

## Inference
```bash
python main.py --task=inference --job=inference --question="What are the benefits of AI in healthcare?" --classifier_model="t5-small" --llm_model="meta-llama/Llama-2-7b-chat-hf"
```