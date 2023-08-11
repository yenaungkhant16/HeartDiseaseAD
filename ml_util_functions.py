import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

# Separates DataFrame column headers into categorical and continuous variables
def separate_column_headers(df, target_column="TenYearCHD"):
    df_rows = df.shape[0]

    categorical_columns = []
    continuous_columns = []

    for column in df.columns:
        if column==target_column:
            continue
        if len(np.unique(df[column])) < 10:  # Threshold for identifying categorical variables (can be adjusted)
            categorical_columns.append(column)
        else:
            continuous_columns.append(column)

    # check number of cols in df matches the return values
    assert(df_rows == (len(categorical_columns) + len(continuous_columns) + len(target_column)))

    return categorical_columns, continuous_columns, target_column

# Displays DataFrame information and summary statistics
def df_info(df):
    # create a dataframe to print the results of:
    # non null values, object type per column, null values
    info_df = pd.DataFrame(columns=['Non-Null Count', 'Data Type'])
    info_df['Non-Null Count'] = df.notnull().sum(axis=0)
    info_df['Data Type'] = df.dtypes
    info_df['Null Count'] = df.isnull().sum()

    # Print the modified info output
    print("########### Display dataframe info ###########")
    print(info_df)
    print("\n########### Display dataframe summary statistics ###########")
    print(df.describe())

# Shows data distribution using countplots and histograms
def plot_data_distribution(df):
    # total number of columns in the data
    num_columns = df.shape[1]
    # number of rows to be printed in the graph
    num_cols_per_row = 4

    # Calculate the number of rows needed
    num_rows = (num_columns + num_cols_per_row - 1) // num_cols_per_row

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols_per_row, figsize=(18, num_rows * 4))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    for i, column in enumerate(df.columns):
        ax = axes[i]
        # This is to filter out the encoded categorical data
        if df[column].nunique() < 10:
            # Categorical data, plot countplot
            sns.countplot(data=df, x=column, ax=ax)
        else:
            # Continuous data, plot histogram with KDE
            sns.histplot(data=df, x=column, kde=True, ax=ax)

        ax.set_title(f'Distribution of {column}')

    # Remove any empty subplots
    if num_columns < num_rows * num_cols_per_row:
        for i in range(num_columns, num_rows * num_cols_per_row):
            fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# Shows target class distribution using a bar plot
def plot_class_distribution(df):
    # Assuming the target column is the last column in the DataFrame
    target_column = df.columns[-1]

    # Create a bar plot using Seaborn
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=target_column)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')

    # Rename the values in target column to "has CHD" and "no CHD"
    plt.xticks(ticks=[0, 1], labels=['Does not have CHD', 'Has CHD'])

    plt.show()


# adapted from chatgpt
# Shows box plots for specified columns
def show_boxplot(df, column_names_for_plot):
    # Calculate the number of rows and columns for the subplots
    num_plots = len(column_names_for_plot)
    num_rows = (num_plots + 2) // 3
    num_cols = min(num_plots, 3)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # Flatten axs if there is only one plot
    if num_rows == 1 and num_cols == 1:
        axs = [axs]

    for i, column in enumerate(column_names_for_plot):
        if i >= num_plots:
            # Hide any extra subplots beyond the required number
            axs[i // num_cols, i % num_cols].axis('off')
        elif column in df.columns:
            ax = axs[i // num_cols, i % num_cols]
            ax.boxplot(df[column].dropna(), vert=False)
            ax.set_title(f'Box Plot of {column}')
            ax.set_xlabel(column)

    plt.tight_layout()
    plt.show()

# Reads a csv file from directory
def read_from_csv(filename, directory):
    # check if this is for current directory
    if (len(directory)>0):
        fpath = os.path.join(directory, filename)
    else:
        fpath = filename
    
    # read the csv file
    try:
        df = pd.read_csv(fpath)
        return df
    except FileNotFoundError:
        print("File not found, created a new dataframe")
        return pd.DataFrame()

# from chatgpt
# Saves the DataFrame to a CSV file
def save_to_csv(df, filename, directory):
    # check if this is for current directory
    if (len(directory)>0):
        fpath = os.path.join(directory, filename)
    else:
        fpath = filename

    if not os.path.exists(directory) and len(directory)>0:
        os.makedirs(directory)

    try:
        df.to_csv(fpath, index=False)
        print("DataFrame successfully saved to CSV.")
    except Exception as e:
        print("Error occurred while saving to CSV:", e)

# Saves the DataFrame to a CSV file
def add_to_csv(df, filename, directory):
    # check if this is for current directory
    if (len(directory)>0):
        fpath = os.path.join(directory, filename)
    else:
        fpath = filename

    # Check if the file already exists
    if os.path.exists(fpath):
        # Load the existing file to compare columns
        existing_df = pd.read_csv(fpath)
        # Check if the columns match
        if not existing_df.columns.equals(df.columns):
            raise ValueError("Columns of the DataFrame do not match the existing CSV file.")
        
        # Append to the existing file (mode='a')
        df.to_csv(fpath, mode='a', header=False, index=False)
    else:
        # Create a new file and save the DataFrame
        df.to_csv(fpath, index=False)
    
# Combines DataFrames by matching columns and padding rows if needed
def combine_df(current_df, new_df):
    # if the column in the new df exists in the current df, remove the columns in the current df
    for col in new_df.columns:
        if col in current_df.columns:
            current_df = current_df.drop(columns=col)

    # if the new df has more rows than  the current df increase size of current df
    if new_df.shape[0] > current_df.shape[0]:
        num_pad_rows = new_df.shape[0] - current_df.shape[0]
        padding_data = pd.DataFrame([[np.nan] * current_df.shape[1]] * num_pad_rows, columns=current_df.columns)
        current_df = pd.concat([current_df, padding_data], ignore_index=True)
    # if current df has more rows than new df, increase size of current df and pad the rows with NaH
    elif new_df.shape[0] < current_df.shape[0]:
        num_pad_rows = current_df.shape[0] - new_df.shape[0]
        padding_data = pd.DataFrame([[np.nan] * new_df.shape[1]] * num_pad_rows, columns=new_df.columns)
        new_df = pd.concat([new_df, padding_data], ignore_index=True)

    return pd.concat([current_df, new_df], axis=1)
