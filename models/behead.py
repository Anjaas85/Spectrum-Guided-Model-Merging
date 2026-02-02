import os
import torch
from transformers import AutoModelForSequenceClassification, AutoModel, AutoConfig

model_names = {
    "sst2": "textattack/bert-base-uncased-SST-2",
    "ag-news": "textattack/bert-base-uncased-ag-news",
    "mnli": "textattack/bert-base-uncased-MNLI",
    "base": "google-bert/bert-base-uncased"
}

# Create directories
os.makedirs("backbones", exist_ok=True)
os.makedirs("heads", exist_ok=True)

for task, path in model_names.items():
    print(f"Processing {task}...")
    
    # 1. Load the model
    if task == "base":
        # The base model is already just a backbone
        model = AutoModel.from_pretrained(path)
        model.save_pretrained(f"backbones/{task}")
    else:
        # Task models have classification heads
        model = AutoModelForSequenceClassification.from_pretrained(path)
        
        # 2. Extract and save the Backbone (the "Beheading")
        # In BERT, the backbone is stored in the 'bert' attribute
        backbone = model.bert 
        backbone.save_pretrained(f"backbones/{task}")
        
        # 3. Extract and save the Head
        # We save the state_dict of the classifier layer
        head_state = model.classifier.state_dict()
        torch.save(head_state, f"heads/{task}_head.pt")
        
        # Also save the config (needed for num_labels)
        model.config.save_pretrained(f"heads/{task}_config")

print("All models beheaded and saved to ./backbones and ./heads")