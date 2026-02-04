import os
import torch
from transformers import AutoModelForSequenceClassification, AutoModel, AutoConfig

model_names = {
    "sst2": "textattack/bert-base-uncased-SST-2",
    "ag-news": "textattack/bert-base-uncased-ag-news",
    "mnli": "textattack/bert-base-uncased-MNLI",
    "base": "google-bert/bert-base-uncased"
}

os.makedirs("backbones", exist_ok=True)

for task, path in model_names.items():
    print(f"Processing {task}...")
    
    if task == "base":
        model = AutoModel.from_pretrained(path)
        model.save_pretrained(f"backbones/{task}")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path)
               
        backbone = model.bert 
        backbone.save_pretrained(f"backbones/{task}")
        

print("All models beheaded")