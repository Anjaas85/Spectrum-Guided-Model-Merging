from datasets import load_dataset

def load_test_dataset(name: str, split: str = None):
    """
    testing part - needed
    """
    name = name.lower()
    valid_names = ["sst2", "ag-news", "mnli"] 
    if name == "cola":
      raise ValueError(f"cola not supported anymore")
    if name not in valid_names:
        raise ValueError(f"Invalid name: {name}, choose one of {valid_names}.")


    if name == "ag-news":
        dataset_dict = load_dataset("fancyzhx/ag_news")
        target_split = split if split else "test"
    
    elif name == "mnli":
        dataset_dict = load_dataset("nyu-mll/glue", "mnli")
        target_split = split if split else "validation_matched"
        
    else:
        dataset_dict = load_dataset("nyu-mll/glue", "sst2")
        target_split = split if split else "validation"

    if target_split not in dataset_dict: #all different have to be checked
        available = list(dataset_dict.keys())
        raise ValueError(
            f"Split '{target_split}' not found in {name.upper()}. "
            f"Available: {available}"
        )

    dataset = dataset_dict[target_split]
    print(f"Loaded {name.upper()} split='{target_split}' with {len(dataset)} samples")
    return dataset


def load_train_val_split(name: str, val_ratio: float = 0.2, seed: int = 42):
    """
    we will make validation split on training set
    in case we are fintuning in the end
    """
    name = name.lower()
    
    if name == "ag-news":
        full_train = load_dataset("ag_news")["train"]
    elif name == "mnli":
        full_train = load_dataset("glue", "mnli")["train"]
    elif name == "sst2":
        full_train = load_dataset("glue", "sst2")["train"]
    else:
        full_train = load_dataset("glue", name)["train"]

    split = full_train.train_test_split(test_size=val_ratio, seed=seed)
    train_data, val_data = split["train"], split["test"]

    print(
        f"Created split for {name.upper()} "
        f"{len(train_data)} train / {len(val_data)} val samples."
    )
    return train_data, val_data