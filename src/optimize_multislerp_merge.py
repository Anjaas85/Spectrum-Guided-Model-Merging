import os
import shutil
import yaml
import copy
import optuna
import torch


from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


from models.unifiedMultiTaskModel import UnifiedMultiTaskModel

# remember to change this if we want the other function
from src.test_modelv2 import evaluate_model
from src.save_results import save_results
from src.data_loader import load_test_dataset

def optimize_multislerp_weights(
    template_config_path, 
    output_dir="./experiments/optuna_multislerp",
    study_name = "multislerp",
    n_trials=20,
    device="cuda"
):

    with open(template_config_path, 'r') as f:
        template_config = yaml.safe_load(f)

    datasets = {}
    tasks = ["sst2", "ag-news", "mnli"]
    for task in tasks:
        try:
            datasets[task] = load_test_dataset(task) #task = dataset name
        except Exception as e:
            print(f"Warning: Could not load dataset {task}: {e}")
            datasets[task] = None 

    os.makedirs(output_dir, exist_ok=True)
    storage_url = f"sqlite:///{os.path.join(output_dir, 'optuna.db')}"
    save_dir = "./experiments/results/optuna_multislerp"

    def objective(trial):
        # suggest parameters
        w_sst2 = 1.0 
        w_agnews_low = trial.suggest_float("w_agnews_low", 0.8, 2.5)
        w_mnli_low = trial.suggest_float("w_mnli_low", 1.5, 4.0)
        w_agnews_high = trial.suggest_float("w_agnews_high", 0.8, 2.5)
        w_mnli_high = trial.suggest_float("w_mnli_high", 1.5, 4.0)

        # config to update
        current_config = copy.deepcopy(template_config)
        
        for i, slice_config in enumerate(current_config['slices']):
            
            sources = slice_config['sources']
            
            for source in sources:
                model_path = source['model'].lower()
                if 'parameters' not in source:
                    source['parameters'] = {}
                    
                if "sst2" in model_path:
                    source['parameters']['weight'] = 1.0                
                elif "ag-news" in model_path or "ag_news" in model_path:
                    # if i ==0 -> low else  high
                    source['parameters']['weight'] = w_agnews_low if i == 0 else w_agnews_high  
                elif "mnli" in model_path:
                    source['parameters']['weight'] = w_mnli_low if i == 0 else w_mnli_high

 

        trial_id = trial.number
        temp_merge_path = os.path.join(output_dir, f"trial_{trial_id}") 
        
        try:
            # merge
            config = MergeConfiguration.model_validate(current_config)
            options = MergeOptions(
                cuda=(device == "cuda"),
                copy_tokenizer=True,
                allow_crimes=True,
                quiet=True 
            )
            run_merge(config, temp_merge_path, options=options)
            
            # evaluate
            multi_model = UnifiedMultiTaskModel(temp_merge_path)
            
            multi_model.add_task_head("sst2", "textattack/bert-base-uncased-SST-2")
            multi_model.add_task_head("ag-news", "textattack/bert-base-uncased-ag-news")
            multi_model.add_task_head("mnli", "textattack/bert-base-uncased-MNLI")
            
            print(f"\n--- Evaluating Trial {trial_id} ---")
            
            res_sst2 = evaluate_model(multi_model, f"trial_{trial_id}", "sst2", device=device, dataset=datasets.get("sst2"))
            res_agnews = evaluate_model(multi_model, f"trial_{trial_id}", "ag-news", device=device, dataset=datasets.get("ag-news"))
            res_mnli = evaluate_model(multi_model, f"trial_{trial_id}", "mnli", device=device, dataset=datasets.get("mnli"))

            acc_sst2 = res_sst2["metrics"]["accuracy"]
            acc_agnews = res_agnews["metrics"]["accuracy"]
            acc_mnli = res_mnli["metrics"]["accuracy"]


            if acc_sst2 < 0.7 or acc_agnews < 0.7 or acc_mnli < 0.65:
              score = 0
            else:
              score = (acc_sst2 + acc_agnews + acc_mnli) 
              #save_results(results, accuracy, dataset_name, model_name, save_dir="experiments/results"):
 
              save_results(res_sst2, acc_sst2, "sst2",f"trial_{trial_id}",save_dir)
              save_results(res_agnews, acc_agnews, "ag-news",f"trial_{trial_id}",save_dir)
              save_results(res_mnli, acc_mnli, "mnli",f"trial_{trial_id}",save_dir)
            
            print(f"[Trial {trial_id} Result] w_AG_low={w_agnews_low:.2f}, w_MNLI_low={w_mnli_low:.2f}, w_AG_high={w_agnews_high:.2f}, w_MNLI_high={w_mnli_high:.2f} | Score: {score:.4f}")

            return score

        except Exception as e:
            print(f"[Trial {trial_id}] Failed: {e}")
            return 0.0
        finally:
            if os.path.exists(temp_merge_path):
                shutil.rmtree(temp_merge_path)

    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,    
        study_name=study_name,  
        load_if_exists=True     
    )

    #if colab breaks
    remaining_trials = n_trials - len(study.trials)
    
    if remaining_trials > 0:
        print(f"remaining trials: '{remaining_trials}'")
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print("All done :)")

    print("Best parameters found:", study.best_params)
    
    return study.best_params