import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import AutoTokenizer


from src.data_loader import load_test_dataset
from src.save_results import save_results

def evaluate_model(
    model_instance,
    model_name: str,
    dataset_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    dataset_name = dataset_name.lower()
    
    # tokeniser - we were adding it to be also with model
    merged_path = os.path.join("experiments/merges", model_name)
    if os.path.exists(merged_path):
        tokenizer = AutoTokenizer.from_pretrained(merged_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    # data
    dataset = load_test_dataset(dataset_name)
    
    true_labels = []
    pred_labels = []
    confidences = []
    
    model_instance.to(device)
    model_instance.eval()

    print(f"Starting evaluation: {model_name} on {dataset_name}...")

    #we need to do differently for each dataset cause different key names
    #especially for mnli cause it uses pairs
    for ex in tqdm(dataset):
        
        text_b = None
        if dataset_name == "sst2":
            text_a = ex["sentence"]
        elif dataset_name == "ag-news":
            text_a = ex["text"]
        elif dataset_name == "mnli":
            text_a = ex["premise"]
            text_b = ex["hypothesis"]
        else:
            text_a = ex["sentence"] if "sentence" in ex else ex["text"]
        
        label = ex["label"]

        inputs = tokenizer(
            text_a, 
            text_pair=text_b, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # we should check if it this is good
        with torch.no_grad():
            logits = model_instance(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                token_type_ids=inputs['token_type_ids'], # <--- ADD THIS
                task_name=dataset_name
            )            
            probs = F.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            
            pred_val = pred.item()
            conf_val = conf.item()

            #MNLI - inconsistancy :(
            if dataset_name == "mnli":
                # textattack    0:contr 1:entail 2:neutral
                # glue: `````````0:entail 1:neutral 2:Contrad
                # we have to rotate (textattack + 2) % 3
                mnli_map = {
                    0: 2, 
                    1: 0, 
                    2: 1  
                }
                pred_val = mnli_map.get(pred_val, pred_val)
    


            true_labels.append(label)
            pred_labels.append(pred_val)
            confidences.append(conf_val)
        

    #metrics part
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    confidences = np.array(confidences)

    acc = accuracy_score(true_labels, pred_labels)
    prec, rec, f1, support = precision_recall_fscore_support(true_labels, pred_labels, average=None)
    
    per_class_stats = {}
    for class_id in np.unique(true_labels):
        class_mask = (true_labels == class_id)
        class_acc = accuracy_score(true_labels[class_mask], pred_labels[class_mask])
        
        pred_mask = (pred_labels == class_id)
        mean_conf = confidences[pred_mask].mean() if pred_mask.any() else 0.0
        
        per_class_stats[int(class_id)] = {
            "accuracy": float(class_acc),
            "precision": float(prec[class_id]),
            "recall": float(rec[class_id]),
            "f1": float(f1[class_id]),
            "mean_confidence": float(mean_conf)
        }

    #macro metr
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro')

    results = {
        "true_labels": true_labels.tolist(),
        "pred_labels": pred_labels.tolist(),
        "confidences": confidences.tolist(),
        "per_class_stats": per_class_stats,
        "metrics": {
            "accuracy": float(acc),
            "f1_macro": float(macro_f1),
            "precision_macro": float(macro_prec),
            "recall_macro": float(macro_rec),
            "confusion_matrix": confusion_matrix(true_labels, pred_labels).tolist()
        }
    }

    results_short = {
        "per_class_stats": per_class_stats,
        "metrics": {
            "accuracy": float(acc),
            "f1_macro": float(macro_f1),
            "precision_macro": float(macro_prec),
            "recall_macro": float(macro_rec),
            "confusion_matrix": confusion_matrix(true_labels, pred_labels).tolist()
        }
    }



    #save everything 
    save_results(results, acc, dataset_name, model_name)
    
    print(f"\nResults for {model_name} on {dataset_name}:")
    print(f"Accuracy: {acc:.4f}     F1: {macro_f1:.4f}")
    return results_short

