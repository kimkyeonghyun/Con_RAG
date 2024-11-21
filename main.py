import time
import argparse

from utils.arguments import ArgParser
from utils.utils import check_path, set_random_seed

def main(args: argparse.Namespace) -> None:
    # Set random seed
    if args.seed is not None:
        set_random_seed(args.seed)

    start_time = time.time()

    # 경로 존재 확인
    for path in []:
        check_path(path)

    # 할 job 얻기
    if args.job is None:
        raise ValueError('Please specify the job to do.')
    else:
        if args.task == 'data_preprocessing':
            if args.job == 'preprocessing':
                from task.preprocessing.data_preprocessing import preprocessing as job
            elif args.job == 'question_classification':
                from task.preprocessing.question_classification import question_classification as job
            elif args.job == 'retriever':
                from task.preprocessing.retriever import retriever as job
        elif args.task == 'classification_training':
            if args.job == 'preprocessing':
                from task.classification_training.question_preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.classification_training.train import training as job
            elif args.job == 'testing':
                from task.classification_training.test import testing as job
        elif args.task == 'classification_evaluation':
            if args.job == 'generation_test':
                from task.classification_evaluation.generation_test import classifier_llm_generation as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'inference':  # Inference task 추가
            if args.job == 'inference':
                from task.inference.inference import inference as job  # inference.py의 job 함수 호출
            else:
                raise ValueError(f'Invalid job: {args.job}')
        else:
            raise ValueError(f'Invalid task: {args.task}')
        
    # job 실행
    job(args)

    elapsed_time = time.time() - start_time
    print(f'Completed {args.job}; Time elapsed: {elapsed_time / 60:.2f} minutes')

if __name__ == '__main__':
    parser = ArgParser()
    args = parser.get_args()

    main(args)