from transformers import AutoTokenizer
import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import fields

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.config import VLMConfig, TrainConfig
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from data.collator import VQACollator

tok = AutoTokenizer.from_pretrained(
    'HuggingFaceTB/SmolLM2-360M-Instruct',
    use_fast=True
)

print(type(tok))
print(tok)