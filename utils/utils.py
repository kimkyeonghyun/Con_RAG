import os
import sys
import time
import tqdm
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F

def check_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_torch_device(device: str):
    if device is not None:
        get_torch_device.device = device

    if 'cuda' in get_torch_device.device:
        if torch.cuda.is_available():
            return torch.device(get_torch_device.device)
        else:
            print('No GPU found. Using CPU.')
            return torch.device('cpu')
    elif 'mps' in device:
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print('MPS not available because the current Pytorch install'
                      ' was not built with MPS enabled.')
                print('Using CPU.')
            else:
                print('MPS not available because the current MacOS version'
                      ' is not 12.3+ and/or you do not have an MPS-enabled'
                      ' device on this machine.')
                print('Using CPU.')
            return torch.device('cpu')
        else:
            return torch.device(get_torch_device.device)
    elif 'cpu' in device:
        return torch.device('cpu')
    else:
        print('No such device found. Using CPU.')
        return torch.device('cpu')

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, 'flush'):
                self.stream.flush()
        finally:
            self.release()
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except Exception:
            self.handleError(record)

def write_log(logger, message):
    if logger:
        logger.info(message)

def get_tb_exp_name(args: argparse.Namespace):
    ts = time.strftime('%Y - %b - %d - %H: %M: %S', time.localtime())
    exp_name = str()
    exp_name += "%s - " % args.task.upper()
    exp_name += "%s - " % args.proj_name

    if args.job in ['training', 'resume_training']:
        exp_name += 'TRAIN - '
        exp_name += 'MODEL = %s - ' % args.model_type.upper()
        exp_name += 'DATA = %s - ' % args.task_dataset.upper()
        exp_name += 'DESC = %s - ' % args.description
    elif args.job == 'testing':
        exp_name += 'TEST - '
        exp_name += 'MODEL = %s - ' % args.model_type.upper()
        exp_name += 'DATA = %s - ' % args.task_dataset.upper()
        exp_name += 'DESC = %s - ' % args.description
    exp_name += 'TS = %s' % ts

    return exp_name

def get_wandb_exp_name(args: argparse.Namespace):
    exp_name = str()
    exp_name += '%s - ' % args.task.upper()
    exp_name += '%s / ' % args.task_dataset.upper()
    exp_name += '%s' % args.model_type.upper()
    return exp_name

def get_huggingface_model_name(model_type: str) -> str:
    name = model_type.lower()
    if name == 't5-small':
        return 'google-t5/t5-small'
    elif name == 't5-base':
        return 'google-t5/t5-base'
    elif name == 't5-large':
        return 'google-t5/t5-large'
    elif name == 'llama2_chat':
        return "meta-llama/Llama-2-7b-chat-hf"

def parse_bool(value: str):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')