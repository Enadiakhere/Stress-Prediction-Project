#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score as f1
from sklearn.dummy import DummyClassifier

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, make_scorer, accuracy_score as acc


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import multiprocessing

multiprocessing.cpu_count()


# In[3]:


start = 1
end = 36


# ### Option 1: Automatic Data Preprocessing

# The function Preprocessing_of_data fetches the raw data from GitHub, processes it, and returns the processed datasets.
# Note: This may take some time. Alternatively, consider Option 2 for a more hands-on approach.

# In[ ]:


# Import necessary libraries and extensions
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from Process_data_def import *


# ### Option 2: Manual Data Preprocessing

# To manually view and run the preprocessing code:
# 1. Ensure the .py file and this Jupyter notebook are in the same directory.
# 2. Uncomment the line below.

# In[ ]:


# %load ./Process_data.py


# ### Set Directory

# In[5]:


directory = r'/Users/henryemagbon/Downloads/DF'
directory = directory.replace("\\", "/")


# ### Optional: Save processed data to CSV

# Adjust the directory path as needed. **Only run this code if you run any of the above data processing code**

# In[4]:


def save_data_to_csv(start=start, end=end, directory=directory):
    for i in range(start, end):
        suffix = f'S{i:02}'
        path = directory + f'df_Stress_{suffix}.csv'
        
        # Saving to CSV
        current_df = globals()[f'df{i}']
        current_df.to_csv(path, index=False)


# In[ ]:


save_data_to_csv()


# ### Read processed data to Variables

# Run this to run load the data even if you have not ran the processing code

# In[6]:


# Import CSV files back for further use
def load_data_from_csv(start=1, end=36, directory=directory):
    for i in range(start, end):
        suffix = f'S{i:02}'
        path = directory + f'/df_Stress_{suffix}.csv'
        
        # Loading from CSV
        globals()[f'df{i}'] = pd.read_csv(path)


# In[7]:


load_data_from_csv(start=1, end=36, directory=directory)


# In[8]:


df12.head(5)


# In[9]:


import random

def generate_random_numbers(n=25, seed=123):
    random.seed(seed)
    available_numbers = list(range(1, 36))  # 1 to 35

    # To ensure n doesn't exceed the length of available numbers
    n = min(n, len(available_numbers))

    result = random.sample(available_numbers, n)
    
    return result

# Example usage:
#print(generate_random_numbers(25))




def df_train_number(length=20):
    numbers = generate_random_numbers(length)
    
    for i, num in enumerate(numbers, 1):
        var_name = f'df{i}'
        if var_name not in globals():  # Check if the variable already exists
            globals()[var_name] = [num]
    
    return numbers


# In[10]:


def custom_round(value):
    fractional_part = value - int(value)
    if fractional_part >= 0.5:
        return int(value) + 1
    else:
        return int(value)


# How many percent do you want to train?

# In[11]:


train_percentage = 83
train_num = custom_round(train_percentage / 100 * (end - 1))


# In[12]:


train_list = df_train_number(train_num)


# In[13]:


def concatenate_dataframes_from_numbers(rand_numbers=train_num):
    dataframes = [globals()[f'df{num}'] for num in rand_numbers]
    return pd.concat(dataframes, axis=0)



df_train = concatenate_dataframes_from_numbers(train_list)


# In[103]:


#df_train


# In[14]:


def df_test_number(num=train_num):
    all_numbers = set(range(start, end-1))
    test_numbers = sorted(all_numbers - set(train_list))
    
    # Create DataFrames for the test numbers
    for num in test_numbers:
        var_name = f'df{num}'
        if var_name not in globals():  # Check if the variable already exists
            globals()[var_name] = pd.DataFrame([num], columns=['Value'])
    
    return test_numbers


# In[15]:


test_list = df_test_number(train_num)


# In[16]:


test_list


# In[17]:


df=df_train.copy()


# In[18]:


for num in test_list:
    # Construct the variable names for X and Y based on the number from test_list
    x_var_name = f"x_df{num}_test"
    y_var_name = f"y_df{num}_test"
    
    # Access the relevant dataframe using the globals function
    source_df = globals()[f"df{num}"]
    
    # Assign the processed data to the newly created variable names
    globals()[x_var_name] = source_df.drop(['Person', 'Label'], axis=1)
    globals()[y_var_name] = source_df['Label']
    
    print("--------------------------------------------------------------------------------")
    print(f"x_df{num}_test, y_df{num}_test = df{num}.drop(['Person','Label'], axis=1), df{num}['Label']")


# In[19]:


for num in test_list:
    print(f'length of S{num} dataframe =', len(globals()[f"df{num}"]))


# In[ ]:





# In[20]:


# Prepare data
x_train, y_train = df.drop(['Person','Label'], axis=1), df['Label']


# In[21]:


def plot_confusion_matrix(y_true, y_pred, ax, title="Confusion Matrix"):
    """Plot the confusion matrix using seaborn and matplotlib."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')


# In[22]:


import time
from IPython.display import display, clear_output

def show_loading_animation(duration=10):
    end_time = time.time() + duration
    while time.time() < end_time:
        for char in ['-', '\\', '|', '/']:
            clear_output(wait=True)
            display(char)
            time.sleep(0.1)

# Test the animation
#show_loading_animation()


# In[ ]:





# In[23]:


def model_analysis(model, x_train, y_train, test_list):
    """Evaluate and visualize model performance."""

    model.fit(x_train, y_train)
    cv_tr_score = cross_val_score(model, x_train, y_train, cv=5, scoring=make_scorer(acc))
    print('Cross-validation score on training data', round(cv_tr_score.mean(), 2))

    CV_scores = []
    FN_values = []
    accuracies = []  # Initialize accuracies list

    # Calculate the number of rows required for the given test_list
    rows = (len(test_list) + 2) // 3

    # Set up figure for confusion matrices
    fig_cm, axes_cm = plt.subplots(nrows=rows, ncols=3, figsize=(15, 5 * rows))

    # Set up figure for classification reports
    fig_report, axes_report = plt.subplots(nrows=rows, ncols=3, figsize=(15, 2 * rows))  # Reduced figure height for lesser spacing

    for index, test_num in enumerate(test_list):
        row = index // 3
        col = index % 3

        x_test = globals()[f'x_df{test_num}_test']
        y_test = globals()[f'y_df{test_num}_test']

        preds = model.predict(x_test)

        # Display accuracy
        accuracy = round(acc(y_test, preds), 2)
        accuracies.append(accuracy)  # Append the accuracy to the accuracies list

        print(f"\nAccuracy of the model on participant S{test_num} data:", accuracy)

        # Cross validate on the test data
        n_ts_cv = cross_val_score(model, x_test, y_test, cv=5, scoring=make_scorer(acc))
        print(f'Cross-validation score on Participant S{test_num} data:', round(n_ts_cv.mean(), 2))
        CV_scores.append(round(n_ts_cv.mean(), 2))

        # Plot confusion matrix
        plot_confusion_matrix(y_test, preds, axes_cm[row, col], title=f'S{test_num} Confusion Matrix')

        # Display classification report
        report = classification_report(y_test, preds)
        axes_report[row, col].axis('off')  # Turn off the axis
        axes_report[row, col].text(0.1, 0.9, f"S{test_num} Classification Report", fontsize=12, weight='bold')  # Repositioned title
        axes_report[row, col].text(0, 0.6, report, fontsize=10, va='top')

        # Append the FN value
        FN_values.append(confusion_matrix(y_test, preds)[1][0])

    # Remove unused subplots
    for i in range(index + 1, 3):
        axes_cm[row, i].axis('off')
        axes_report[row, i].axis('off')

    fig_cm.tight_layout()
    fig_report.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)  # Adjusted padding for lesser spacing

    fig_cm.suptitle('Confusion Matrices', y=1.02)
    fig_report.suptitle('Classification Reports', y=1.05)

    plt.show()

    for i, test_num in enumerate(test_list):
        print(f"\nCV of S{test_num} =", CV_scores[i])
        x_test_global = globals()[f'df{test_num}']
        print(f"S{test_num} Accuracy =", round(acc(x_test_global['Label'], model.predict(x_test_global.drop(['Person', 'Label'], axis=1))), 2))
        print(f"The FN of S{test_num} =", FN_values[i])

    return accuracies, CV_scores, FN_values


# In[24]:


# Define your models and their names
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
}

# Store the results of model_analysis for each model
results = {}

show_loading_animation()
for model_name, model in models.items():
    print(f"\n{model_name} Analysis\n{'-' * 20}")
    results[model_name] = model_analysis(model, x_train, y_train, test_list)


# In[29]:


def plot_metric_values(test_labels, results):
    """Plot accuracy, cross-validation, and FN for multiple models side-by-side."""
    
    # Set up the seaborn style
    sns.set_style("whitegrid")

    metrics = [
        ("Accuracy Scores for Test Datasets", "blue"),
        ("Cross-Validation Scores for Test Datasets", "green"),
        ("False Negatives for Test Datasets", "red")
        # Add more metrics here if needed
    ]
    
    num_metrics = len(metrics)
    num_rows = (num_metrics + 1) // 2 if num_metrics >= 5 else (num_metrics // 2) + (num_metrics % 2)

    plt.figure(figsize=(18, 6 * num_rows))
    
    for index, (title, color) in enumerate(metrics):
        row = index // 2 + 1
        col = index % 2 + 1
        plt.subplot(num_rows, 2, (row-1)*2 + col)

        for model_name, (accuracies, CV_scores, FN_values) in results.items():
            if title.startswith("Accuracy"):
                sns.lineplot(x=test_labels, y=accuracies, marker="o", label=model_name)
            elif title.startswith("Cross-Validation"):
                sns.lineplot(x=test_labels, y=CV_scores, marker="o", label=model_name)
            else:
                sns.lineplot(x=test_labels, y=FN_values, marker="o", label=model_name)

        plt.title(title)
        plt.xlabel("Test Dataset")
        plt.ylabel(title.split(" ")[0])  # Extract the first word of the title for the y-axis
        plt.legend()

    plt.tight_layout()
    plt.show()


# In[30]:


test_labels = [f"S{test_num}" for test_num in test_list]


# In[31]:


plot_metric_values(test_labels, results)


# In[ ]:





# In[32]:


def plot_metric_values(test_labels, results):
    """Plot accuracy, cross-validation, and FN for multiple models side-by-side using bar plots."""
    
    # Set up the seaborn style
    sns.set_style("whitegrid")

    metrics = [
        ("Accuracy Scores for Test Datasets", "blue"),
        ("Cross-Validation Scores for Test Datasets", "green"),
        ("False Negatives for Test Datasets", "red")
        # Add more metrics here if needed
    ]
    
    num_metrics = len(metrics)
    num_rows = (num_metrics + 1) // 2 if num_metrics >= 5 else (num_metrics // 2) + (num_metrics % 2)

    plt.figure(figsize=(18, 6 * num_rows))
    
    for index, (title, _) in enumerate(metrics):  # color from metrics list is no longer used
        row = index // 2 + 1
        col = index % 2 + 1
        plt.subplot(num_rows, 2, (row-1)*2 + col)
        data = []

        for model_name, (accuracies, CV_scores, FN_values) in results.items():
            if title.startswith("Accuracy"):
                data.append(accuracies)
            elif title.startswith("Cross-Validation"):
                data.append(CV_scores)
            else:
                data.append(FN_values)

        # Transpose the data for proper plotting
        data = list(map(list, zip(*data)))

        df = pd.DataFrame(data, columns=list(results.keys()), index=test_labels)

        # Melt the data for seaborn bar plotting
        melted_data = df.melt(value_name='value', var_name='model', ignore_index=False).reset_index().rename(columns={'index': 'Test Dataset'})
        sns.barplot(data=melted_data, x='Test Dataset', y='value', hue='model', palette='tab10')  # Reversed Blues for better distinction

        plt.title(title)
        plt.ylabel(title.split(" ")[0])  # Extract the first word of the title for the y-axis
        plt.legend()

    plt.tight_layout()
    plt.show()


# In[131]:


plot_metric_values(test_labels, results)


# In[ ]:





# In[33]:


def calculate_mean(values):
    """Return the mean of a list of values."""
    return sum(values) / len(values)

# Dictionary to store the mean values for each model and metric
mean_values = {}

for model_name, (accuracies, CV_scores, FN_values) in results.items():
    accuracies_mean = round(calculate_mean(accuracies),2)
    CV_scores_mean = round(calculate_mean(CV_scores),2)
    FN_values_mean = round(calculate_mean(FN_values))

    print(f"{model_name}:\n"
          f"Mean of Accuracies: {accuracies_mean:.2f}\n"
          f"Mean of CV_scores: {CV_scores_mean:.2f}\n"
          f"Mean of FN_values: {FN_values_mean:.2f}\n")

    # Store the mean values in the dictionary
    mean_values[model_name] = (accuracies_mean, CV_scores_mean, FN_values_mean)

# Now, mean_values contains the mean of each metric for each model


# In[34]:


mean_values


# In[35]:


def plot_mean_values(mean_values):
    """Plot mean values of accuracy, cross-validation, and FN for multiple models."""

    sns.set_style("whitegrid")
    metrics = [
        "Mean Accuracy Scores",
        "Mean Cross-Validation Scores",
        "Mean False Negatives"
        # Add more metrics titles here if needed in the future
    ]

    num_metrics = len(metrics)
    num_rows = (num_metrics // 3) + (num_metrics % 3)  # Calculate the number of rows based on number of metrics

    plt.figure(figsize=(18, 6 * num_rows))
    
    for index, title in enumerate(metrics):
        plt.subplot(num_rows, 3, index+1)
        
        # Extracting values for the given metric
        values = [mean_values[model][index] for model in mean_values.keys()]
        
        sns.barplot(x=list(mean_values.keys()), y=values, palette="deep")
        plt.title(title)
        plt.ylabel(title.split(" ")[1])  # Extract the second word of the title for the y-axis
        plt.xticks(rotation=45)  # Rotate model names for better visibility

    plt.tight_layout()
    plt.show()


# In[36]:


# Now you can call the function with your mean_values dictionary
plot_mean_values(mean_values)


# In[ ]:





# In[ ]:





# In[ ]:




