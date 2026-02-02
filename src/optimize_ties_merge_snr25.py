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
from src.test_modelv2 import evaluate_model
from src.data_loader import load_test_dataset
from src.save_results import save_results
def optimize_ties_params(
    template_config_path, 
    output_dir="./experiments/optuna_ties_snr25",
    study_name = "ties",
    n_trials=50,
    device="cuda"
):


    with open(template_config_path, 'r') as f:
        template_config = yaml.safe_load(f)

    datasets = {}
    tasks = ["sst2", "ag-news", "mnli"]
    for task in tasks:
        try:
            datasets[task] = load_test_dataset(task)
        except Exception as e:
            print(f"Warning: Could not load dataset {task}: {e}")
            datasets[task] = None 

    os.makedirs(output_dir, exist_ok=True)
    storage_url = f"sqlite:///{os.path.join(output_dir, 'optuna.db')}"
    save_dir = "./experiments/results/optuna_ties_snr25"

    def objective(trial):
        w_sst2 = 1.0 

        
        w_agnews_high = trial.suggest_float("w_agnews_high", 0.3, 0.9)
        w_mnli_high = trial.suggest_float("w_mnli_high", 0.3, 0.9)
        d_sst2_high = trial.suggest_float("d_sst2_high", 0.15, 0.45)
        d_agnews_high = trial.suggest_float("d_agnews_high", 0.2, 0.45)
        d_mnli_high = trial.suggest_float("d_mnli_high", 0.35, 0.65)

        current_config = copy.deepcopy(template_config)
        
        for i, slice_config in enumerate(current_config['slices']):
            sources = slice_config['sources']
            for source in sources:
                model_path = source['model'].lower()
                if 'parameters' not in source:
                    source['parameters'] = {}
                
                if "sst2" in model_path:
                    source['parameters']['weight'] = w_sst2
                    source['parameters']['density'] = d_sst2_high
                elif "ag-news" in model_path or "ag_news" in model_path:
                    source['parameters']['weight'] =  w_agnews_high
                    source['parameters']['density'] =  d_agnews_high
                elif "mnli" in model_path:
                    source['parameters']['weight'] = w_mnli_high
                    source['parameters']['density'] =  d_mnli_high
                elif "base" in model_path:
                    if i == 1:
                        source['parameters']['weight'] = 0.00
                    else: source['parameters']['weight'] = 1.00

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
            score = 1
            res_sst2 = evaluate_model(multi_model, f"trial_{trial_id}", "sst2", device=device, dataset=datasets.get("sst2"))
            acc_sst2 = res_sst2["metrics"]["accuracy"]
            print(acc_sst2)
            if acc_sst2 < 0.6:
              score = 0 
            else:
              res_agnews = evaluate_model(multi_model, f"trial_{trial_id}", "ag-news", device=device, dataset=datasets.get("ag-news"))
              acc_agnews = res_agnews["metrics"]["accuracy"]
              print(acc_agnews)
              if acc_agnews < 0.7:
                score = 0
              else:
                res_mnli = evaluate_model(multi_model, f"trial_{trial_id}", "mnli", device=device, dataset=datasets.get("mnli"))
                acc_mnli = res_mnli["metrics"]["accuracy"]
                print(acc_mnli)
                if acc_mnli < 0.4:
                  score = 0
                else:
                  score = ((acc_sst2 + acc_agnews + 5 * acc_mnli)/2)
                  save_results(res_sst2, acc_sst2, "sst2",f"trial_{trial_id}",save_dir)
                  save_results(res_agnews, acc_agnews, "ag-news",f"trial_{trial_id}",save_dir)
                  save_results(res_mnli, acc_mnli, "mnli",f"trial_{trial_id}",save_dir)
            
            
            print(f"[Trial {trial_id} Result] , w_AG_high={w_agnews_high:.2f}, w_MNLI_high={w_mnli_high:.2f}")
            print(f" , d_AG_high={d_agnews_high:.2f}, d_MNLI_high={d_mnli_high:.2f} |")
            print(f" d_sst2_high={d_sst2_high:.2f}||||| Score: {score:.4f}")
            return score

        except Exception as e:
            print(f"[Trial {trial_id}] Failed: {e}")
            return 0.0
        finally:
            if os.path.exists(temp_merge_path):
                shutil.rmtree(temp_merge_path)

    study = optuna.create_study(
      direction="maximize",
      storage = storage_url,
      study_name = study_name,
      load_if_exists = True
    )

    remaining_trials = n_trials - len(study.trials)

    if remaining_trials > 0:
        print(f"remaining trials: '{remaining_trials}'")
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print("All done :)")

    print("Best parameters found:", study.best_params)
    
    return study.best_params