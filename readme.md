# Machine Learning Model README

This README file provides instructions for using and understanding the machine learning model developed for the Framingham Heart Study dataset. The model aims to predict heart disease risk based on various features collected from the participants.

## Installation and Dependencies

For the Python version and required libraries, please refer to the document "installation.txt" included in this repository. Ensure you have the necessary Python environment set up before proceeding.

## Code Files

The project consists of several Jupyter Notebook files and Python modules. Here's a brief overview of each file:

1. `00_FinalModel.ipynb`: The final model for training after data processing, feature engineering, and model tuning.

2. `01_DataProcessing.ipynb`: Performs data processing on the raw dataset, including handling missing values and data cleaning.

3. `02_FeatureSelection.ipynb`: Selects the relevant features to be used in model training, based on analysis and importance.

4. `03_FeatureExtraction.ipynb`: Performs feature extraction to generate additional useful features if applicable.

5. `04_ModelTuning.ipynb`: Conducts model tuning using a combination of random search (RandomizedSearchCV) and grid search (GridSearchCV) for hyperparameter optimization.

6. `05_SelectBestModel.ipynb`: After obtaining the model hyperparameters from model tuning, this notebook applies them to their respective models and selects the best-performing model.

7. `Appendix_A-Feature_Selection_Extraction.ipynb`: Provides the results of the feature selection and feature extraction processes and prints out the results.

8. `ml_functions.py`: Contains the custom functions utilized in the Jupyter Notebooks for model training and evaluation.

9. `ml_util_functions.py`: Holds utility functions that support the main machine learning operations.

## Dataset

The dataset "Framingham.csv" contains the data collected for the Framingham Heart Study. To understand the dataset structure and the variables involved, please refer to the provided research paper available at: [Research Paper Link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9416695/)

## Folders

This project includes the following folders:

1. `csv_files`: This folder stores any data extracted during training or relevant files imported by the code files. However, please note that the main dataset is not stored here.

2. `models`: Contains the trained machine learning models, scaler objects, and data imputer used during the training process.

3. `notebook_pdf`: Contains PDF versions of the Jupyter Notebooks for easy reference.
