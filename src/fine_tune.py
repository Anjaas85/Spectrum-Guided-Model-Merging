import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import copy

from src.data_loader import load_train_val_split
from src.test_modelv2 import evaluate_model # or testmodel v1??

def get_active_layers_from_yaml(yaml_path):
    """
    Parses a mergekit YAML file to find which layers were involved in the merge.
    Assumes mergekit syntax where 'layer_range' is [start, end) (exclusive).
    Returns a set of integer layer indices.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Merge config not found at: {yaml_path}")
        
    print(f"Parsing merge config: {yaml_path}")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    active_layers = set()
    
    if 'slices' in config:
        for sl in config['slices']:
            if 'sources' in sl:
                for source in sl['sources']:
                    if 'layer_range' in source:
                        start, end = source['layer_range']
                        for i in range(start, end):
                            active_layers.add(i)

    if not active_layers and 'slices' not in config:
        print("Warning: No 'slices' found in YAML. Assuming all layers active.")
        return set(range(12)) #bert backbone layers???
        
    print(f"Identified Spectrum-Selected Active Layers: {sorted(list(active_layers))}")
    return active_layers

def configure_model_freezing(model, active_layer_indices):
    """
    Freeze entire model except classification heads, layers selected with Spectrum (found in config), LayerNorm params, pooling
    """
    for param in model.parameters():
        param.requires_grad = False
        
    for name, param in model.named_parameters():
        if "heads" in name or "classifier" in name:
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

# LiNeS / LLRD (high lr at top, low at bottom)
def get_lines_optimizer_params(model, head_lr, decay_rate=0.9):
    """
    Implements LiNeS (Layer-wise Learning Rate Decay).
    Layer 11 (Top) gets head_lr. Layer 0 (Bottom) gets head_lr * decay^11.
    """
    optimizer_grouped_parameters = []
    head_params = []
    backbone_layer_params = {} # layer_idx -> list
    #embeddings_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "heads" in name:
            head_params.append(param)
        elif "encoder.layer." in name:
            try:
                parts = name.split("encoder.layer.")
                layer_idx = int(parts[1].split(".")[0])
                if layer_idx not in backbone_layer_params:
                    backbone_layer_params[layer_idx] = []
                backbone_layer_params[layer_idx].append(param)
            except:
                other_params.append(param)
        #elif "embeddings" in name:
        #    embeddings_params.append(param)
        else:
            other_params.append(param)

    if head_params:
        optimizer_grouped_parameters.append({"params": head_params, "lr": head_lr})
    
    num_layers = 12 #bert backbone
    for layer_i in range(num_layers):
      # layer_i=11 -> decay_power=0 -> lr=head_lr
      # layer_i=0  -> decay_power=11 -> lr=head_lr * decay^11
      decay_power = (num_layers - 1) - layer_i
      layer_lr = head_lr * (decay_rate ** decay_power)
        
      if layer_i in backbone_layer_params:
          optimizer_grouped_parameters.append({
              "params": backbone_layer_params[layer_i],
              "lr": layer_lr
          })
    if other_params:
        optimizer_grouped_parameters.append({
            "params": other_params,
            "lr": head_lr * (decay_rate ** num_layers)
        })
    
    return optimizer_grouped_parameters

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
    def __init__(self, tasks, ref_losses, alpha=2.0, smoothing=0.1):
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
          scores.append(max(ratio, 0.001) ** self.alpha) # add small epsilon to avoid 0, although unlikely with EMA
        total_score = sum(scores)
        if total_score == 0:
            probs = [1.0/len(self.tasks)] * len(self.tasks)
        else:
            probs = [s / total_score for s in scores]
            
        return np.random.choice(self.tasks, p=probs)


def fine_tune_and_evaluate(
    multitask_model_instance,
    model_name: str,
    #merge_config_path: str, 
    #tasks=["sst2", "mnli", "ag-news"],
    #epochs=3,
    total_steps = 2000,
    head_lr=2e-5,
    decay_rate=0.9,
    batch_size=16,
    device="cuda"
):  
    new_model_name = f"{model_name}_ft"
    print(f"\nStarting post-merge finetuning for {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    multitask_model_instance.to(device)
   
    merge_config_path = f"./experiments/merges/{model_name}/mergekit_config.yml"
    active_layers = get_active_layers_from_yaml(merge_config_path)
    configure_model_freezing(multitask_model_instance, active_layers)

    tasks=["sst2", "mnli", "ag-news"]
    # loading 10% data
    train_dataloaders = {}
    for task in tasks:
        train_data, _ = load_train_val_split(task, val_ratio=0.1) 
        train_dataloaders[task] = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    task_iterators = {
        t: get_infinite_batch_iterator(train_dataloaders[t], t, tokenizer, device) 
        for t in tasks
    }

    #setup scheduler
    # theoretical max entropy = -ln(1/N) = ln(N)
    ref_losses = {
        "sst2": np.log(2),      # 0.69
        "mnli": np.log(3),      # 1.10
        "ag-news": np.log(4)    # 1.39
    }

    current_refs = {t: ref_losses.get(t, 1.0) for t in tasks}

    scheduler_sampler = DynamicTaskScheduler(tasks, current_refs, alpha=2.0)

    grouped_params = get_lines_optimizer_params(multitask_model_instance, head_lr, decay_rate)
    optimizer = AdamW(grouped_params)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1*total_steps), #num_warmup_steps=int(0.1 * total_steps_per_epoch * epochs)
        num_training_steps=total_steps #num_training_steps=total_steps_per_epoch * epochs
    )

    print(f"Starting training for {total_steps} steps...")
    multitask_model_instance.train()
    progress_bar = tqdm(range(total_steps))
    for step in progress_bar:
        # A. Sample Task
        current_task = scheduler_sampler.sample_task()
        
        # B. Get Batch
        inputs, labels = next(task_iterators[current_task])
        
        # C. Forward & Backward
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
        scheduler.step()
        scheduler_sampler.update(current_task, loss.item())

        '''
        mnli_ratio = scheduler_sampler.current_losses['mnli'] / ref_losses['mnli']
        progress_bar.set_postfix({
            "task": current_task, 
            "loss": f"{loss.item():.3f}", 
            "MNLI_Ratio": f"{mnli_ratio:.2f}" 
        })
        '''
    #save finetuned model
    save_path = f"experiments/merges/{new_model_name}" #do we want it here or in models/finetuned??
    os.makedirs(save_path, exist_ok=True)
    multitask_model_instance.encoder.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    #save finetuned heads!!!
    os.makedirs(os.path.join(save_path, "heads"), exist_ok=True)
    for task_name, head_module in multitask_model_instance.heads.items():
      torch.save(head_module.state_dict(), os.path.join(save_path, "heads", f"{task_name}_head.pt"))
    print(f"Fine-tuned model saved")
    
    #final evaluation
    print(f"Running final evaluation ...")
    results = {}
    for task in tasks:
        res = evaluate_model(multitask_model_instance, new_model_name, task, device=device)
        results[task] = res["metrics"]["accuracy"]
        
    print(f"Post-fine-tuning results: {results}")
    return results