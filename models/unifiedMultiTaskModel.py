import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification

class UnifiedMultiTaskModel(nn.Module):
    def __init__(self, backbone_path):
        super().__init__()
        # backbone
        self.encoder = AutoModel.from_pretrained(backbone_path)
        self.heads = nn.ModuleDict()

    def add_task_head(self, task_name, original_model_path):
        orig = AutoModelForSequenceClassification.from_pretrained(original_model_path)

        # we are adding also pooler here - bcs mergekit is mergin up to layer 12
        self.heads[task_name] = nn.Sequential(
            orig.bert.pooler,
            nn.Dropout(0.1),
            orig.classifier
        )
        print(f"Added head for: {task_name}")

    def forward(self, input_ids, attention_mask, token_type_ids=None, task_name=None):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids  # this is important for mnli to catch conection between two sentence (for sst2 and ag-news not neccessary)
        )

       
        logits = self.heads[task_name](outputs[0])

        return logits