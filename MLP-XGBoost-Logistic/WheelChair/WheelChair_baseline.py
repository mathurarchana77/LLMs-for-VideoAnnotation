#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 13:44:20 2025

@author: nmit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 12:32:09 2025
   
@author: nmit
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


import warnings

warnings.filterwarnings("ignore")


#from supervised_models import mlp, logit, xgb_model
#from utils import perf_metric, print_results
#from preprocessing import load_dataset



from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import openpyxl


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_dataset(dataset_path, target_column, label_data_rate=0.1, save_scaler=True , feature_cols=None, test_size=0.2):
    """
    Loads and preprocesses a physiological dataset for emotion classification.

    Args:
        dataset_path (str): Path to the CSV dataset file.
        target_column (str): Column name of the target variable (e.g., 'class1' or 'class2').
        feature_cols (list): List of feature column names to use. If None, defaults to CASE dataset features.
        label_data_rate (float): Fraction of training data to treat as labelled (for semi-supervised setups).
        test_size (float): Proportion of data reserved for testing.
        save_scaler (bool): Whether to fit and save a new scaler or load an existing one.

    Returns:
        x_train (np.ndarray): Labelled training features
        y_train (np.ndarray): Labelled training labels
        x_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        x_unlab (np.ndarray): Unlabelled training features
    """
    # load case dataset
    df = pd.read_csv(dataset_path)

    if feature_cols is None:
        feature_cols = ['acc_x_mean','acc_x_var','acc_y_mean','acc_y_var','acc_z_mean','acc_z_var',
 'acc_sum_mean','acc_abssum_mean','acc_sum_var','acc_abssum_var','acc_maxabssum',
 'gyr_x_mean','gyr_x_var','gyr_y_mean','gyr_y_var','gyr_z_mean','gyr_z_var',
 'gyr_sum_mean','gyr_abssum_mean','gyr_sum_var','gyr_abssum_var','gyr_maxabssum']

    # one-hot encode class column
    ohe = OneHotEncoder()

    df_ohe = pd.DataFrame(ohe.fit_transform(df[[target_column]]).toarray())
    df_ohe.columns = [f"{target_column}_{i}" for i in range(df_ohe.shape[1])]
    df = pd.concat([df, df_ohe], axis=1)

    label_cols = list(df_ohe.columns)
    df = df[feature_cols + label_cols]

    X = df[feature_cols]
    Y = df[label_cols]

    # feature scaling
    if save_scaler:
        scaler = MinMaxScaler()
        X[feature_cols] = scaler.fit_transform(X[feature_cols])
        joblib.dump(scaler, "scaler.pkl")
        print("✅ Scaler fitted and saved.")
    else:
        # 🆕 Load the existing scaler and use transform instead of fit_transform
        scaler = joblib.load("scaler.pkl")
        X[feature_cols] = scaler.transform(X[feature_cols])
        print("✅ Loaded existing scaler and applied same transformation.")

    """
    Train-test split
    20% :- Test 
    10% of 80% = 8% :- Labelled dataset
    90% of 80% = 72% :- Unlabelled dataset
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42, stratify=Y
    )

    # converting to numpy arrays
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values

    # Divide labeled and unlabeled data
    idx = np.random.permutation(len(y_train))

    # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    # Unlabeled data
    x_unlab = x_train[unlab_idx, :]

    # Labeled data
    x_train = x_train[label_idx, :]
    y_train = y_train[label_idx, :]

    return x_train, y_train, x_test, y_test, x_unlab


# -------------------------------------------------------------
# 1️⃣ Label Conversion Utilities
# -------------------------------------------------------------
def convert_matrix_to_vector(matrix):
    """Convert a one-hot encoded matrix into a 1D vector of class labels."""
    no, dim = matrix.shape
    vector = np.zeros([no, ])
    for i in range(dim):
        idx = np.where(matrix[:, i] == 1)
        vector[idx] = i
    return vector


def convert_vector_to_matrix(vector):
    """Convert a 1D vector of labels into a one-hot encoded matrix."""
    no = len(vector)
    dim = len(np.unique(vector))
    matrix = np.zeros([no, dim])
    for i in range(dim):
        idx = np.where(vector == i)
        matrix[idx, i] = 1
    return matrix

# -------------------------------------------------------------
# 2️⃣ Performance Metrics & Evaluation
# -------------------------------------------------------------

def perf_metric(metric, y_test, y_test_hat):
    """
    Evaluate model performance.

    Args:
        metric: 'acc', 'auc', or 'f1'
        y_test: true labels (one-hot)
        y_test_hat: predicted probabilities

    Returns:
        performance: computed metric score
    """
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_test_hat, axis=1)

    if metric == 'acc':
        return accuracy_score(y_true, y_pred)
    elif metric == 'auc':
        return roc_auc_score(y_test[:, 1], y_test_hat[:, 1])
    elif metric == 'f1':
        return f1_score(y_true, y_pred, average='weighted')
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def print_results(acc, auc, f1=None, model_name="Model"):
    """Print mean and standard deviation of metrics for a model."""
    print(f"\n📊 {model_name} Results")
    print(f"Accuracy: {np.mean(acc):.4f} ± {np.std(acc):.4f}")
    print(f"AUC: {np.mean(auc):.4f} ± {np.std(auc):.4f}")
    if f1 is not None:
        print(f"F1-score: {np.mean(f1):.4f} ± {np.std(f1):.4f}")

#%%%
def mask_generator(p_m, x):
    """Generate mask vector.
  
  Args:
    - p_m: corruption probability
    - x: feature matrix
    
  Returns:
    - mask: binary mask matrix 
  """
    mask = np.random.binomial(1, p_m, x.shape)
    return mask


#%%
def pretext_generator(m, x):
    """Generate corrupted samples.
  
  Args:
    m: mask matrix
    x: feature matrix
    
  Returns:
    m_new: final mask matrix after corruption
    x_tilde: corrupted feature matrix
  """

    # Parameters
    no, dim = x.shape

    # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]  #along each column shuffle the data across samples.

    # Corrupt samples
    x_tilde = x * (1 - m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde

# -------------------------------------------------------------
# 4️⃣ Train/Test Splitting & Labeled-Unlabeled Division
# -------------------------------------------------------------
def split_dataset(X, y, feature_cols, label_data_rate=0.1, test_size=0.2):
    """
    Generic preprocessing, scaling, train-test splitting, and labeled/unlabeled split.

    Args:
        X: feature DataFrame
        y: one-hot encoded labels
        feature_cols: list of feature column names to scale
        label_data_rate: proportion of labeled data in training
        test_size: test data proportion

    Returns:
        x_lab, y_lab, x_unlab, x_test, y_test
    """
    # Scaling
    scaler = MinMaxScaler()
    X[feature_cols] = scaler.fit_transform(X[feature_cols])

    # Split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=33)

    # Convert to numpy
    x_train, x_test = x_train.values, x_test.values
    y_train, y_test = y_train.values, y_test.values

    # Labeled / Unlabeled split
    idx = np.random.permutation(len(y_train))
    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    x_lab, y_lab = x_train[label_idx], y_train[label_idx]
    x_unlab = x_train[unlab_idx]

    return x_lab, y_lab, x_unlab, x_test, y_test

# -------------------------------------------------------------
# Results Saving (Optional)
# -------------------------------------------------------------

def append_to_excel(acc, auc, filepath='ablation_studies.xlsx'):
    """Append accuracy and AUC results to an Excel file."""
    workbook = openpyxl.load_workbook(filepath)
    worksheet = workbook.active

    max_length = max(len(acc), len(auc))
    for i in range(max_length):
        value1 = acc[i] if i < len(acc) else ''
        value2 = auc[i] if i < len(auc) else ''
        worksheet.append([value1, value2])

    worksheet.append([" ", " "])
    workbook.save(filepath)
#%%
def one_hot_encode_wheelchair(df, class_label):
    """Converts class labels of wheelchair dataset to one-hot encoded vectors.

    Args:
      - df: dataframe containing data
      - class_label: class label

    Returns:
      - X: extracted relevant features
      - y: extracted relevant one-hot encoded labels
    """
    #one-hot encode class column
    ohe = OneHotEncoder()
    df_ohe = pd.DataFrame(ohe.fit_transform(df[[class_label]]).toarray())
    df = df.join(df_ohe)

    df.drop('class', axis=1, inplace=True)
    X = df.loc[:, :'gyr_maxabssum']
    y = df.iloc[:, 22:]
    return X, y





def one_hot_encode_kemocon(df_X, df_y, class_label):
    """Converts class labels of k-EmoCon dataset to one-hot encoded vectors.

  Args:
    - df_X: dataframe containing features
    -df_y: dataframe containing labels
    -class_label: class label

  Returns:
    - X: extracted relevant features
    - y: extracted relevant one-hot encoded labels
  """
    ohe = OneHotEncoder()
    # One-hot encoded labels
    df_ohe = pd.DataFrame(ohe.fit_transform(df_y[[class_label]]).toarray())
    df = df_y.join(df_ohe)

    #extracting relevant features and labels
    X = df_X.iloc[:, 1:]
    y = df.iloc[:, 3:].values
    return X, y

def print_wheelchair_results(acc, auc, f1score, config):
    """Prints list of final accuracy and AUC scores. Also prints beautifully mean and standard deviation
     valence or arousal accuracy and AUC scores after rounding to 4 decimal places.

     Args:
       - acc: accuracy list
       - auc: AUC list
       - label: either 'valence' or 'arousal'
       - config: combination of loss function and activation function such as (logcosh + Param Elliot)
     """
    print(np.round(acc, 4))
    print(np.round(auc, 4))
    print(np.round(f1score, 4))
    print(f"Mean of accuracies for {config} is: {round(np.mean(acc), 4)}")
    # Compute standard deviation
    std_dev_acc = np.std(acc)
    print(f"Standard Deviation of Accuracies using {config}: {round(std_dev_acc, 4)}")

    print(f"Mean of AUC Score for {config} is {round(np.mean(auc), 4)}")
    std_dev_auc = np.std(auc)
    print(f"Standard Deviation of AUC scores using {config}: {round(std_dev_auc, 4)}")

    print(f"Mean of F1 Score for {config} is {round(np.mean(f1score), 4)}")
    std_dev_f1 = np.std(f1score)
    print(f"Standard Deviation of AUC scores using {config}: {round(std_dev_f1, 4)}")

#%%
def wheelchair_results_excel(acc, auc, f1):
  """Append accuracy and AUC to Excel file.

  Args:
    - acc: accuracy score
    - auc: AUC score
  """

  # Load the existing Excel workbook
  workbook = openpyxl.load_workbook('ablation studies.xlsx')

  # Select the active worksheet
  worksheet = workbook.active

  # Determine the maximum length among the three lists
  max_length = max(len(acc), len(auc))

  # Iterate over the range of maximum length
  for i in range(max_length):
    # Get values from each list or empty string if index out of range
    value1 = acc[i] if i < len(acc) else ''
    value2 = auc[i] if i < len(auc) else ''
    value3 = f1[i] if i < len(f1) else ''

    # Write values to Excel
    worksheet.append([value1, value2, value3])

  # Append an empty row
  worksheet.append([" ", " ", " "])

  # Save the workbook
  workbook.save('ablation studies.xlsx')
#%%
def return_5_median(list_acc, list_auc):
    median_list = list(zip(list_acc, list_auc))
    # Sort the list based on the first column
    sorted_list = sorted(median_list, key=lambda x: x[0])

    # Calculate the index range for the middle five elements
    middle_index_start = (len(sorted_list) - 5) // 2
    middle_index_end = middle_index_start + 5

    # Extract ACC and AUC values for the middle five elements
    median_acc = [item[0] for item in sorted_list[middle_index_start:middle_index_end]]
    median_auc = [item[1] for item in sorted_list[middle_index_start:middle_index_end]]

    # Print or use median_acc and median_auc as required
    print("Median ACC values:", median_acc)
    print("Median AUC values:", median_auc)
    return median_acc, median_auc


    
def logit(x_train, y_train, x_test):
  """Logistic Regression

  Args:
    - x_train, y_train: training dataset
    - x_test: testing feature

  Returns:
    - y_test_hat: predicted values for x_test
  """
  # Convert labels into proper format
  if len(y_train.shape) > 1:
    y_train = convert_matrix_to_vector(y_train)  
  
  # Define and fit model on training dataset
  model = LogisticRegression()
  model.fit(x_train, y_train)
  
  # Predict on x_test
  y_test_hat = model.predict_proba(x_test) 
  
  return y_test_hat

#%%
def xgb_model(x_train, y_train, x_test):
  """XGBoost.

  Args:
    - x_train, y_train: training dataset
    - x_test: testing feature

  Returns:
    - y_test_hat: predicted values for x_test
  """
  # Convert labels into proper format
  if len(y_train.shape) > 1:
    y_train = convert_matrix_to_vector(y_train)

  # Define and fit model on training dataset
  model = xgb.XGBClassifier()
  model.fit(x_train, y_train)

  # Predict on x_test
  y_test_hat = model.predict_proba(x_test)

  return y_test_hat


#%%
def mlp(x_train, y_train, x_test, parameters=None):
  """Multi-layer perceptron (MLP).

  Args:
    - x_train, y_train: training dataset
    - x_test: testing feature
    - parameters: hidden_dim, epochs, activation, batch_size, num_layers

  Returns:
    - y_test_hat: predicted values for x_test
  """
  if parameters is None:
    parameters = {
      "hidden_dim": 100,
      "epochs": 100,
      "activation": "relu",
      "batch_size": 128,
      "num_layers": 1
    }

  # Convert labels into proper format
  if len(y_train.shape) == 1:
    y_train = convert_vector_to_matrix(y_train)

  # Divide training and validation sets (9:1)
  idx = np.random.permutation(len(x_train))
  train_idx = idx[:int(len(idx)*0.9)]
  valid_idx = idx[int(len(idx)*0.9):]

  # Validation set
  x_valid = x_train[valid_idx, :]
  y_valid = y_train[valid_idx, :]

  # Training set
  x_train = x_train[train_idx, :]
  y_train = y_train[train_idx, :]

  # Reset the graph
  K.clear_session()

  # Define network parameters
  hidden_dim = parameters['hidden_dim']
  epochs_size = parameters['epochs']
  act_fn = parameters['activation']
  batch_size = parameters['batch_size']

  # Define basic parameters
  data_dim = x_train.shape[1]
  label_dim = y_train.shape[1]

  # Build model - with number of layers specifed by parameters
  model = Sequential()
  model.add(Dense(hidden_dim, input_dim = data_dim, activation = act_fn))
  for i in range(0, parameters['num_layers']):
    model.add(Dense(hidden_dim, activation = act_fn))
  model.add(Dense(label_dim, activation = 'softmax'))

  model.compile(loss = 'categorical_crossentropy', optimizer='adam',
                metrics = ['acc'])

  es = EarlyStopping(monitor='val_loss', mode = 'min',
                     verbose = 1, restore_best_weights=True, patience=50)

  # Fit model on training dataset
  model.fit(x_train, y_train, validation_data = (x_valid, y_valid),
            epochs = epochs_size, batch_size = batch_size,
            verbose = 0, callbacks=[es])

  # Predict on x_test
  y_test_hat = model.predict(x_test)
  #print("SLAB MLP was trained on the test.")
  return y_test_hat

MODEL_REGISTRY = {
    "logit": logit,
    "xgb": xgb_model,
    "mlp": mlp
}



# Quick test
if __name__ == "__main__":
    x_train, y_train, x_test, y_test, x_unlab = load_dataset(
        dataset_path="wheelchair.csv",
        target_column="class",
        feature_cols=None,
        label_data_rate=0.1,
        test_size=0.2)
    print("✅ Preprocessing complete!")
    print("Train shape:", x_train.shape)
    print("Test shape:", x_test.shape)
    print("Unlabelled shape:", x_unlab.shape)
    
    
# 1️⃣ Load preprocessed data
dataset_path = "wheelchair.csv"
target_column = "class"
x_train, y_train, x_test, y_test, x_unlab = load_dataset(dataset_path, target_column, save_scaler=False)

# 2️⃣ Define MLP hyperparameters (same as original)
mlp_parameters = {
    'hidden_dim': 100,
    'epochs': 100,
    'activation': 'relu',
    'batch_size': 128,
    'num_layers': 1
}

# 3️⃣ Unified training function for both models
def run_model(model_name, x_train, y_train, x_test, y_test, runs=5, mlp_params=None):
    acc_list, auc_list, f1_list = [], [], []

    for i in range(runs):
        # Choose model
        if model_name == "logit":
            y_test_hat = logit(x_train, y_train, x_test)
        elif model_name == "mlp":
           y_test_hat = mlp(x_train, y_train, x_test, mlp_params)
        elif model_name == "xgb":
            y_test_hat = xgb_model(x_train, y_train, x_test)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Metrics
        acc = perf_metric("acc", y_test, y_test_hat)
        auc = perf_metric("auc", y_test, y_test_hat)
        f1 = perf_metric("f1", y_test, y_test_hat)

        acc_list.append(acc)
        auc_list.append(auc)
        f1_list.append(f1)

        print(f"✅ {model_name.upper()} | Run {i+1}: Acc = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}")

    # Final summary
    print_results(acc_list, auc_list, f1_list, model_name=model_name.upper())

# 4️⃣ Run both models
print("\n🚀 Running Logistic Regression...")
run_model("logit", x_train, y_train, x_test, y_test, runs=5)

print("\n🚀 Running MLP...")
run_model("mlp", x_train, y_train, x_test, y_test, runs=5, mlp_params=mlp_parameters)

print("\n🚀 Running XGBoost...")
run_model("xgb", x_train, y_train, x_test, y_test, runs=5)



