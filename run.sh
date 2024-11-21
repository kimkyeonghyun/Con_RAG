DATASET=naturalqa
BS=32
LR=5e-5
EP=10
DEVICE=cuda
clear
MODEL=t5-small
LLM=llama2_chat

# Preprocessing, Retrieval, and Contribution-Based Question Classification for Training
python main.py --task=preprocessing --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=preprocessing --job=retriever --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=preprocessing --job=question_classification --task_dataset=${DATASET} --model_type=${MODEL} --llm_model=${LLM}

# Train and test the classification model
python main.py --task=classification_training --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification_training --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}
python main.py --task=classification_training --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}

# Perform generation testing using the classification model
python main.py --task=classification_evaluation --job=generation_test --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE} --llm_model=${LLM}
