from transformers import AutoTokenizer
import os

# Define your local backbone paths
backbone_paths = [
    "backbones/base",
    "backbones/sst2",
    "backbones/ag-news",
    "backbones/mnli"
]

# Load the tokenizer (using the base BERT as the source)
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

for path in backbone_paths:
    if os.path.exists(path):
        print(f"Saving tokenizer to {path}...")
        tokenizer.save_pretrained(path)
    else:
        print(f"⚠️ Directory not found: {path}")

print("\nDone! All backbones now have tokenizer files. Mergekit will be able to copy them now.")