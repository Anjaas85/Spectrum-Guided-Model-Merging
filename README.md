# Spectrum-Guided Model Merging

## Overview

Model merging techniques allow combining expert models without retraining. Standard wholemodel merging may underperform due to interference of noisy or conflicting updates, so we propose Spectrum-guided model merging to identify and merge only high-informative layers. We employ 3 BERT-base models fine-tuned on SST-2, MNLI, and AG-News and we compare selective merging (Top 25% and 50% highSNR layers) against standard merging, experimenting with both TIES and MultiSLERP algorithms. Our results show that merging just the top 50% and 25% of layers via TIES retains 99.2% and 92.7% of full-merge performance. After a lightweight post-merge fine-tuning (only on the selected layers), we managed to achieve respectively 96.7% and 95.4% of the original models’ accuracy. This confirms that focusing on highSNR layers enables efficient multi-task model merging.


## Repository Structure
You can find all the information on how to run our code in SpectrumGUidedModelMerging_finalNotebook.ipnyb Jupyter notebook. When it comes to structure:
```

├── experiment/              # Optuna trials databases
│   ├── merges/              # Optimal weight configurations
├── models/                  # Scripts to prepare beheaded models
│   ├──finetuned_weights/    # Post-merge fine-tuned weights
├── configs                  # configurations for models to be merged
└── src/                     # Source code implementation
```

When it comes to calculaiton of SNR significant layers it can be found in Compute_Layer_SNR_with_Spectrum.ipynb.
