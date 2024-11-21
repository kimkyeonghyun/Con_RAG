import os
import sys
import argparse
import torch
import torch.nn as nn
from transformers import T5ForSequenceClassification, T5Config
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 
from utils.utils import get_huggingface_model_name

class ClassificationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ClassificationModel, self).__init__()
        self.args = args

        
        model_name = get_huggingface_model_name(args.model_type)
        config = T5Config.from_pretrained(model_name)
        self.model = T5ForSequenceClassification.from_pretrained(model_name, config=config)
    def forward(self, question, labels):
            
        output = self.model(question, decoder_input_ids = question, labels = labels)

        return output.logits