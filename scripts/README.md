# cicada-gsoc-evaluation-task

## Scripts

- `scripts/dataset_loader.py`: Contains functions for loading and preparing datasets for training and evaluation. The `TrainingDatasetLoader` class is used for loading the dataset, splitting it into training and test sets, and returning TensorFlow datasets for model training.
- `scripts/exploration_plots.py`: Functions for plotting exploratory data visualizations like PCA, pixel-label patterns, and image combinations for label visualization.
- `scripts/model_evaluation.py`: Contains functions for evaluating model performance, including metrics like ROC curve, confusion matrix, precision-recall curve, and more.
- `scripts/model_plots.py`: Functions for visualizing model predictions and training performance, including loss, accuracy, and visual model architecture.
- `scripts/model.py`: Defines an autoencoder model class and a classifier model class for evaluation tasks.
- `scripts/utils.py`: Utility functions for saving and loading parameters using YAML files.
