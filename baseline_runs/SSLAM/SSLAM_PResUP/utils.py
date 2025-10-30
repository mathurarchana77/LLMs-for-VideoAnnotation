"""
Various utility functions 
(1) mask_generator: Generate mask vector for self and semi-supervised learning
(2) pretext_generator: Generate corrupted samples for self and semi-supervised learning
(3) perf_metric: prediction performances in terms of AUROC or accuracy
(4) convert_matrix_to_vector: Convert two-dimensional matrix into one dimensional vector
(5) convert_vector_to_matrix: Convert one dimensional vector into one dimensional matrix
"""

# Necessary packages
import openpyxl
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


#%%
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


#%%
def perf_metric(metric, y_test, y_test_hat):
    """Evaluate performance.
  
  Args:
    - metric: acc or auc
    - y_test: ground truth label
    - y_test_hat: predicted values
    
  Returns:
    - performance: Accuracy or AUC-ROC performance
  """
    # Accuracy metric
    if metric == 'acc':
        result = accuracy_score(np.argmax(y_test, axis=1),
                                 np.argmax(y_test_hat, axis=1))
    
    # AUROC metric
    elif metric == 'auc':
         result = roc_auc_score(y_test[:, 1], y_test_hat[:, 1])
    
    # F1 Score metric (binary classification)
    elif metric == 'f1':
        result = f1_score(np.argmax(y_test, axis=1),
                           np.argmax(y_test_hat, axis=1),
                           average='macro')
    return result


#%%
def convert_matrix_to_vector(matrix):
    """Convert two dimensional matrix into one dimensional vector.
  
  Args:
    - matrix: two-dimensional matrix
    
  Returns:
    - vector: one-dimensional vector
  """
    # Parameters
    no, dim = matrix.shape

    # Define output
    vector = np.zeros([no, ])

    # Convert matrix to vector
    for i in range(dim):
        idx = np.where(matrix[:, i] == 1)
        vector[idx] = i

    return vector


#%%
def convert_vector_to_matrix(vector):
    """Convert one dimensional vector into two dimensional matrix
  
  Args:
    - vector: one dimensional vector
    
  Returns:
    - matrix: two dimensional matrix
  """
    # Parameters
    no = len(vector)
    dim = len(np.unique(vector))
    # Define output
    matrix = np.zeros([no, dim])

    # Convert vector to matrix
    for i in range(dim):
        idx = np.where(vector == i)
        matrix[idx, i] = 1

    return matrix


# %%
def one_hot_encode_case(df, class_label):
    """Converts class labels of CASE dataset to one-hot encoded vectors.

    Args:
      - df: dataframe containing data
      - class_label: class label

    Returns:
      - X: extracted relevant features
      - y: extracted relevant one-hot encoded labels
    """
    #one-hot encode class column
    ohe = OneHotEncoder()
    #Choose either class1 or class2 to select either valence or arousal
    df_ohe = pd.DataFrame(ohe.fit_transform(df[[class_label]]).toarray())
    df = df.join(df_ohe)

    df.drop('class1', axis=1, inplace=True)
    df.drop('class2', axis=1, inplace=True)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.drop('valence', axis=1, inplace=True)
    df.drop('arousal', axis=1, inplace=True)

    X = df.loc[:, :'emg_trap']
    y = df.iloc[:, 8:]
    return X, y

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

#%%
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

# %%
def split_train_to_unlabel_labeled(x_train, y_train, label_data_rate):
    """Split training data into unlabeled and labeled data sets.

    Args:
    - x_train: training data
    - y_train: training labels
    - label_data_rate: percentage of labeled data

    Returns:
    - x_unlab: unlabeled data
    - x_lab: labeled data
    - y_lab: labeled labels
    """
    # Divide labeled and unlabeled data
    idx = np.random.permutation(len(y_train))

    # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    # Unlabeled data
    x_unlab = x_train[unlab_idx, :]

    # Labeled data
    x_lab = x_train[label_idx, :]
    y_lab = y_train[label_idx, :]
    return x_unlab, x_lab, y_lab

# %%
def case_split(X, y, label_data_rate, test_size=0.2):
    """Preprocess dataset and split into train and test set. Then further split train dataset into
          unlabeled and labeled set.

    Args:
    - X: extracted relevant features
    - y: extracted relevant one-hot encoded labels
    - test_size: size of test set

    Returns:
    - X_unlab, X_train, X_test, y_train, y_test
    """
    # Preprocessing dataset
    norm_scaler = MinMaxScaler()
    X[['ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']] = norm_scaler.fit_transform(
        X[['ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']])

    std_scaler = StandardScaler()
    X[['ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']] = std_scaler.fit_transform(
        X[['ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']])

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=33)

    # converting to numpy arrays
    x_train = x_train.iloc[:, :].values
    y_train = y_train.iloc[:, :].values
    x_test = x_test.iloc[:, :].values
    y_test = y_test.iloc[:, :].values
    x_train.shape, x_test.shape, y_train.shape, y_test.shape

    x_unlab, x_train, y_train = split_train_to_unlabel_labeled(x_train, y_train, label_data_rate)
    
    return x_unlab, x_train, x_test, y_train, y_test

#%%
def kemocon_split(X, y, test_size=0.2, label_data_rate=0.1):
    """Preprocess dataset and split into train and test set. Then further split train dataset into
      unlabeled and labeled set.

    Args:
    - X: extracted relevant features
    - y: extracted relevant one-hot encoded labels
    -test_size: size of test set

    Returns:
    - X_unlab, X_train, X_test, y_train, y_test
    """
    #Preprocessing dataset
    normalizer = MinMaxScaler()
    X = normalizer.fit_transform(X)

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=33)

    x_unlab, x_train, y_train = split_train_to_unlabel_labeled(x_train, y_train, label_data_rate)

    return x_unlab, x_train, x_test, y_train, y_test

#%%
def wheelchair_split(X, y, label_data_rate, test_size=0.2):
    """Preprocess dataset and split into train and test set. Then further split train dataset into
          unlabeled and labeled set.

    Args:
    - X: extracted relevant features
    - y: extracted relevant one-hot encoded labels
    - test_size: size of test set

    Returns:
    - X_unlab, X_train, X_test, y_train, y_test
    """
    # Preprocessing dataset
    norm_scaler = MinMaxScaler()
    X[['acc_x_mean', 'acc_x_var', 'acc_y_mean', 'acc_y_var', 'acc_z_mean', 'acc_z_var', 'acc_sum_mean',
       'acc_abssum_mean', 'acc_sum_var', 'acc_abssum_var', 'acc_maxabssum', 'gyr_x_mean', 'gyr_x_var', 'gyr_y_mean',
       'gyr_y_var', 'gyr_z_mean', 'gyr_z_var', 'gyr_sum_mean', 'gyr_abssum_mean', 'gyr_sum_var', 'gyr_abssum_var',
       'gyr_maxabssum']] = norm_scaler.fit_transform(X[['acc_x_mean', 'acc_x_var', 'acc_y_mean', 'acc_y_var',
                                                        'acc_z_mean', 'acc_z_var', 'acc_sum_mean', 'acc_abssum_mean',
                                                        'acc_sum_var', 'acc_abssum_var', 'acc_maxabssum', 'gyr_x_mean',
                                                        'gyr_x_var', 'gyr_y_mean', 'gyr_y_var', 'gyr_z_mean',
                                                        'gyr_z_var', 'gyr_sum_mean', 'gyr_abssum_mean', 'gyr_sum_var',
                                                        'gyr_abssum_var', 'gyr_maxabssum']])

    std_scaler = StandardScaler()
    X[['acc_x_mean', 'acc_x_var', 'acc_y_mean', 'acc_y_var', 'acc_z_mean', 'acc_z_var', 'acc_sum_mean',
       'acc_abssum_mean', 'acc_sum_var', 'acc_abssum_var', 'acc_maxabssum', 'gyr_x_mean', 'gyr_x_var', 'gyr_y_mean',
       'gyr_y_var', 'gyr_z_mean', 'gyr_z_var', 'gyr_sum_mean', 'gyr_abssum_mean', 'gyr_sum_var', 'gyr_abssum_var',
       'gyr_maxabssum']] = std_scaler.fit_transform(X[['acc_x_mean', 'acc_x_var', 'acc_y_mean', 'acc_y_var',
                                                       'acc_z_mean', 'acc_z_var', 'acc_sum_mean', 'acc_abssum_mean',
                                                       'acc_sum_var', 'acc_abssum_var', 'acc_maxabssum', 'gyr_x_mean',
                                                       'gyr_x_var', 'gyr_y_mean', 'gyr_y_var', 'gyr_z_mean',
                                                       'gyr_z_var', 'gyr_sum_mean', 'gyr_abssum_mean', 'gyr_sum_var',
                                                       'gyr_abssum_var', 'gyr_maxabssum']])

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=33)

    # converting to numpy arrays
    x_train = x_train.iloc[:, :].values
    y_train = y_train.iloc[:, :].values
    x_test = x_test.iloc[:, :].values
    y_test = y_test.iloc[:, :].values

    x_unlab, x_train, y_train = split_train_to_unlabel_labeled(x_train, y_train, label_data_rate)

    return x_unlab, x_train, x_test, y_train, y_test

#%%
def append_to_excel(acc, auc):
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

    # Write values to Excel
    worksheet.append([value1, value2])

  # Append an empty row
  worksheet.append([" ", " "])

  # Save the workbook
  workbook.save('ablation studies.xlsx')

#%%
def print_results(acc, auc, label, config):
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
    print(f"Mean of {label} accuracies for {config} is: {round(np.mean(acc), 4)}")

    # Compute standard deviation
    std_dev_acc = np.std(acc)
    print(f"Standard Deviation of {label} Accuracies using {config}: {round(std_dev_acc, 4)}")

    print(f"Mean of {label} accuracies AUC Score for {config} is {round(np.mean(auc), 4)}")
    std_dev_auc = np.std(auc)
    print(f"Standard Deviation of {label} AUC scores using {config}: {round(std_dev_auc, 4)}")

#%%
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
