This is all the code related to the Master's thesis _Reference Free Multidimensional Evaluation of Customer Service Conversations_.

The repository includes the following folders and files
- `/model`: All files related to the model used throughout the thesis.
- `/notebooks`: Notebooks used for data preprocessing and data visualization.
- `/scripts`: SLURM scrips to run different jobs on Idun, NTNUs HPC-cluster.
- `bootstrap_corr_test.py`: Code to run the bootstrapping test used in the _Results and Analysis_ section of thesis.
- `dialog_discrimination_dataset.py`: The pre-training dataset in the form of a PyG InMemoryDataset.
- `dialog_rating_dataset.py`: The fine-tuning dataset in the form of a PyG InMemoryDataset.
- `memory_profiling.py`: Code to run memory profiling
- `model_manager.py`: Helper class to train models and different loss functions.
- `pre_training.py`: Code used to run the pre-training process.
- `utils.py`: Small utility functions.
