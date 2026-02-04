import os
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel
from tqdm import tqdm
import copy

from src.data_loader import load_train_val_split
from src.test_modelv2 import evaluate_model as eval_model # or testmodel v1??
from src.test_model import evaluate_model as eval_and_save

SNR25_LAYERS = [8, 9, 10]
SNR50_LAYERS = [0, 2, 7, 8, 9, 10]
 
def get_baseline_performance(task_name):
    """
    Loads the baseline performance (Accuracy or F1/MCC) of the original models.
    Adjust paths if your directory structure differs.
    """
    task_map = {
        "sst2": "sst2/sst2_sst2.json",
        "ag-news": "ag-news/ag-news_ag-news.json",
        "mnli": "mnli/mnli_mnli.json"
    }
    
    base_path = "./experiments/results"
    full_path = os.path.join(base_path, task_map[task_name])
    
    if not os.path.exists(full_path):
        print(f"Warning: Baseline for {task_name} not found at {full_path}. Assuming 1.0 (no scaling).")
        return 0.90 # Fallback placeholder
        
    with open(full_path, 'r') as f:
        data = json.load(f)
        # Prefer Accuracy, fallback to macro avg f1 if needed
        return data["metrics"]["accuracy"]

def apply_spectrum_freezing(model, active_layer_indices):
    """
    Freeze entire model except classification heads, layers selected with Spectrum (found in config), LayerNorm params, pooling
    """
    for param in model.parameters():
        param.requires_grad = False
        
    for name, param in model.named_parameters():
        if "heads" in name:
            param.requires_grad = True
        elif "LayerNorm" in name:
            param.requires_grad = True
        elif "pooler" in name:
            param.requires_grad = True
        elif "encoder.layer." in name:
            try:
                #bert.encoder.layer.X.
                parts = name.split("encoder.layer.")
                layer_part = parts[1].split(".")[0]
                layer_idx = int(layer_part)
                
                if layer_idx in active_layer_indices:
                    param.requires_grad = True
            except:
                pass

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model Configured: {trainable/1e6:.2f}M trainable params / {total/1e6:.2f}M total")

#dynamic sampler helper
def get_infinite_batch_iterator(dataloader, task_name, tokenizer, device):
    """Yields batches indefinitely for a specific task."""
    iterator = iter(dataloader)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        if task_name == "sst2":
            text_a = batch["sentence"]
            text_b = None
        elif task_name == "ag-news":
            text_a = batch["text"]
            text_b = None
        elif task_name == "mnli":
            text_a = batch["premise"]
            text_b = batch["hypothesis"]
        
        labels = batch['label'].to(device)
        inputs = tokenizer(
            text_a, text_pair=text_b,
            padding=True, truncation=True, max_length=128,
            return_tensors="pt"
        ).to(device)
        
        yield inputs, labels

class DynamicTaskScheduler:
    """
    Manages task sampling probabilities proportional to RELATIVE loss.
    P(task) ~ (Current_Loss / Reference_Loss) ^ alpha
    Reference Loss = Theoretical random guess loss (ln(num_classes)).
    """
    def __init__(self, tasks, ref_losses, alpha=1.0, smoothing=0.1):
        self.tasks = tasks
        self.ref_losses = ref_losses
        self.alpha = alpha 
        self.smoothing = smoothing
        self.current_losses = {t: ref_losses[t] for t in tasks}
        
    def update(self, task, loss_val):
        # exponential moving average (smooth out batch noise)
        self.current_losses[task] = (1 - self.smoothing) * self.current_losses[task] + self.smoothing * loss_val
        
    def sample_task(self):
      # calculate scores: (current / ref)^alpha
        scores = []
        for t in self.tasks:
          ratio = self.current_losses[t] / self.ref_losses[t]
          #scores.append(max(ratio, 0.001) ** self.alpha) # add small epsilon to avoid 0, although unlikely with EMA
          scores.append(ratio ** self.alpha)
        total_score = sum(scores)
        if total_score == 0:
            probs = [1.0/len(self.tasks)] * len(self.tasks)
        else:
            probs = [s / total_score for s in scores]
            
        return np.random.choice(self.tasks, p=probs)


def fine_tune_and_evaluate(
    multitask_model_instance,
    model_name: str,
    output_run_name: str = None,
    total_steps = 2000, 
    base_lr=2e-5,
    batch_size=16,
    device="cuda",
    #increase val interval if more than 1 epoch
    val_check_interval=200, # check ARR every X steps 
    arr_threshold=0.9, #stop if ARR reaches this
    patience=4,
    train_subset_ratio=0.1, #use 10% data
):  
    print(f"\nStarting post-merge finetuning for {model_name}")

    if output_run_name is None:
        output_run_name = f"{model_name}_ft"

    if "snr25" in model_name:
        active_layers = SNR25_LAYERS
    elif "snr50" in model_name:
        active_layers = SNR50_LAYERS
    else:
        print("Warning: model name does not contain 'snr25' or 'snr50'")
        active_layers = SNR25_LAYERS 

    apply_spectrum_freezing(multitask_model_instance, active_layers)
    multitask_model_instance.to(device)

    hyperparams = {
        "original_model": model_name,
        "run_name": output_run_name,
        "active_layers": active_layers,
        "total_steps": total_steps,
        "base_lr": base_lr,
        "batch_size": batch_size,
        "val_check_interval": val_check_interval,
        "arr_threshold": arr_threshold,
        "patience": patience,
        "train_subset_ratio": train_subset_ratio,
        "optimizer": "AdamW"
    }
    results_dir = os.path.join("experiments/results", output_run_name)
    os.makedirs(results_dir, exist_ok=True)
    hp_path = os.path.join(results_dir, "hyperparameters.json")
    with open(hp_path, "w") as f:
        json.dump(hyperparams, f, indent=4)
    print(f"Hyperparameters saved to {hp_path}")

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tasks=["sst2", "mnli", "ag-news"]
    # loading 10% data
    train_dataloaders = {}
    val_datasets = {}
    for task in tasks:
        #load standard split (90% train, 10% val)
        train_data, val_data = load_train_val_split(task, val_ratio=0.1) 
        val_datasets[task] = val_data

        #select 10% of training data
        num_samples = len(train_data)
        subset_size = int(num_samples * train_subset_ratio)
        # Randomly select indices
        indices = torch.randperm(num_samples)[:subset_size].tolist()
        train_data_subset = train_data.select(indices)

        train_dataloaders[task] = DataLoader(train_data_subset, batch_size=batch_size, shuffle=True)
    
    task_iterators = {
        t: get_infinite_batch_iterator(train_dataloaders[t], t, tokenizer, device) 
        for t in tasks
    }

    #setup scheduler
    # theoretical loss floors = -ln(1/N) = ln(N)
    ref_losses = {
        "sst2": np.log(2),      # 0.69
        "mnli": np.log(3),      # 1.10
        "ag-news": np.log(4)    # 1.39
    }

    #current_refs = {t: ref_losses.get(t, 1.0) for t in tasks}

    scheduler_sampler = DynamicTaskScheduler(tasks, ref_losses, alpha=1.0)

    params_to_train = []
    for p in multitask_model_instance.parameters():
        if p.requires_grad:
            params_to_train.append(p)
    optimizer = AdamW(params_to_train, lr=base_lr)

    #load baselines for ARR
    baselines = {t: get_baseline_performance(t) for t in tasks}

    save_path_root = "./models/finetuned_models"
    save_path_ckpt = os.path.join(save_path_root, output_run_name)

    #training loop
    print(f"Starting training for {total_steps} steps...")

    best_arr = 0.0
    patience_counter = 0
    start_step = 0
    
    #restart from checkpoint
    state_path = os.path.join(save_path_ckpt, "training_state.json")
    opt_path = os.path.join(save_path_ckpt, "optimizer.pt")

    if os.path.exists(state_path):
        print(f"found checkpoint at {state_path}, resuming...")
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
                start_step = state["step"]
                best_arr = state["best_arr"]
                patience_counter = state.get("patience_counter", 0)
      
            # Load weights
            multitask_model_instance.encoder = AutoModel.from_pretrained(save_path_ckpt)
            heads_dir = os.path.join(save_path_ckpt, "heads")
            for task_name, head_module in multitask_model_instance.heads.items():
                head_path = os.path.join(heads_dir, f"{task_name}_head.pt")
                if os.path.exists(head_path):
                    head_module.load_state_dict(torch.load(head_path))
            
            if os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path))
                
            multitask_model_instance.to(device)
            print(f"Resumed from step {start_step} with Best ARR {best_arr:.4f}")

        except Exception as e:
            print(f"Failed to resume: {e}. Starting from scratch.")
            start_step = 0
            best_arr = 0.0   
            patience_counter = 0   
    
    multitask_model_instance.train()

    progress_bar = tqdm(range(start_step, total_steps))
    for step in progress_bar:
        # sample Task
        current_task = scheduler_sampler.sample_task()
        
        # get batch
        inputs, labels = next(task_iterators[current_task])

        # remap labels: glue (0=ent, 1=neu, 2=con), textattack (0=con, 1=ent, 2=neu) 
        #map 0->2, 1->2, 2->0
        if current_task == "mnli":
            labels = torch.tensor([(l.item() + 1) % 3 for l in labels], device=device)
                
        # fw and bw
        optimizer.zero_grad()
        logits = multitask_model_instance(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs.get('token_type_ids'),
            task_name=current_task
        )
            
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(multitask_model_instance.parameters(), 1.0)
        
        optimizer.step()
        scheduler_sampler.update(current_task, loss.item())

        #validation and early stopping
        if (step + 1) % val_check_interval == 0:
            print(f"\n[Step {step+1}] Running Intermediate Validation...")
            
            # Note: evaluate_model uses the official test/val sets. 
            # This might be slow. If too slow, consider evaluating on a subset.
            current_metrics = {}
            for t in tasks: 
                # Suppress print inside loop to keep clean
                # this should be done on val split, this function is doing on test split
                res = eval_model(
                    multitask_model_instance, 
                    output_run_name, 
                    t, 
                    device=device,
                    dataset=val_datasets[t]
                )
                current_metrics[t] = res["metrics"]["accuracy"]
                
            # calculate ARR
            ratios = [current_metrics[t] / baselines[t] for t in tasks]
            arr = sum(ratios) / len(ratios)
            
            print(f"ARR: {arr:.4f} | Ratios: SST2={ratios[0]:.2f}, MNLI={ratios[1]:.2f}, AG={ratios[2]:.2f}")
            
            # save if best
            if arr > best_arr+0.01:
                best_arr = arr
                patience_counter = 0
                os.makedirs(save_path_ckpt, exist_ok=True)
                # save backbone
                multitask_model_instance.encoder.save_pretrained(save_path_ckpt)
                tokenizer.save_pretrained(save_path_ckpt)
                # save heads!!
                heads_dir = os.path.join(save_path_ckpt, "heads")
                os.makedirs(heads_dir, exist_ok=True)
                for task_name, head_module in multitask_model_instance.heads.items():
                    torch.save(head_module.state_dict(), os.path.join(heads_dir, f"{task_name}_head.pt"))
                
                torch.save(optimizer.state_dict(), opt_path)
                # save state
                with open(state_path, "w") as f:
                    json.dump({
                        "step": step + 1, 
                        "best_arr": best_arr,
                        "patience_counter": patience_counter
                    }, f)
            else:
                patience_counter += 1
            # early stopping
            if arr >= arr_threshold:
                print(f"Early stopping triggered: ARR {arr:.4f} >= {arr_threshold}")
                break
            if patience_counter >= patience:
                print(f"Early stopping triggered: ARR plateaued for {patience_counter} checks.")
                break
                
            multitask_model_instance.train() # Switch back to train mode


        '''
        mnli_ratio = scheduler_sampler.current_losses['mnli'] / ref_losses['mnli']
        progress_bar.set_postfix({
            "task": current_task, 
            "loss": f"{loss.item():.3f}", 
            "MNLI_Ratio": f"{mnli_ratio:.2f}" 
        })
        '''

    print(f"Fine-tuned model saved to {save_path_ckpt}")
    
    #final evaluation
    print(f"Running final evaluation on best checkpoint...")
    #reload model
    multitask_model_instance.encoder = AutoModel.from_pretrained(save_path_ckpt)
    heads_dir = os.path.join(save_path_ckpt, "heads")
    for task_name, head_module in multitask_model_instance.heads.items():
        head_path = os.path.join(heads_dir, f"{task_name}_head.pt")
        if os.path.exists(head_path):
            head_module.load_state_dict(torch.load(head_path))

    multitask_model_instance.to(device)
    multitask_model_instance.eval()
    
    results = {}
    for task in tasks:
        res = eval_and_save(multitask_model_instance, output_run_name, task, device=device) #is it correct to pass model name like that or save_path_ckpt needed?
        results[task] = res["metrics"]["accuracy"]
        
    print(f"Post-fine-tuning results (ARR={best_arr:.4f}): {results}")
    return results