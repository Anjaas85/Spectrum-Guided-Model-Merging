import os
import json
from datetime import datetime
import numpy as np

#we want to save in json files
#but also keep one summary file
#so that we can check easily which model is which

def save_results(results, accuracy, dataset_name, model_name, save_dir="experiments/results"):
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    out_path = os.path.join(model_dir, f"{model_name}_{dataset_name}.json")
    summary_path = os.path.join(save_dir, "summary.txt")


    output_data = {
        "model": model_name,
        "dataset": dataset_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_accuracy": float(accuracy),
        "metrics": results["metrics"],
        "per_class_stats": results["per_class_stats"],
    }


    
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    


    #summaryy
    with open(summary_path, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Model: {model_name} | "
                f"Dataset: {dataset_name} | Acc: {accuracy:.4f} | "
                f"F1-Macro: {results['metrics']['f1_macro']:.4f}\n")

    print(f"Saved detailed results to {out_path}")
    print(f"Updated summary")