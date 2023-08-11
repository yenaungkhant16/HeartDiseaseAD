import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import KernelPCA
from skpp import ProjectionPursuitRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import make_scorer, recall_score, confusion_matrix
from sklearn.impute import MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor


# Removes outliers from specified columns of the input DataFrame using the IQR method
def preprocessing_remove_outliers_iqr(df_original, col_with_outliers_names, multiplier=1.5):
    # Create a copy of the DataFrame to avoid modifying the original
    df = df_original.copy()

    for col_name in col_with_outliers_names:
        # calculate IQR, ignore NaN
        q1 = np.nanpercentile(df[col_name], 25, method='midpoint')
        q3 = np.nanpercentile(df[col_name], 75, method='midpoint')
        iqr = q3 - q1

        # calculate the lower and upper bounds using the multiplier
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        # filter out values outside of the range
        df = df[(df[col_name] >= lower) & (df[col_name] <= upper)]

    # reset the index
    df.reset_index(drop=True, inplace=True)

    # Result validation
    assert df.shape[0] <= df_original.shape[0], "Dataframe should not be bigger after removing outliers"

    return df

# Removes outliers from specific columns based on visual confirmation from boxplots
def remove_outliers_by_boxplot(df):
    # track number of rows for validation
    original_df_rows = df.shape[0]

    # filter out values outside of the range
    df = df[(df["totChol"] <= 550)]
    df = df[(df["sysBP"] <= 275)]
    df = df[(df["BMI"] <= 50)]
    df = df[(df["glucose"] <= 350)]

    # reset the index
    df.reset_index(drop=True, inplace=True)

    # Result validation
    assert df.shape[0] <= original_df_rows, "Dataframe should not be bigger after removing outliers"

    return df

# Rounds and enforces categorical values to stay within specified ranges
def enforce_categorical_values(df_categorical, categorical_data_range):
    # track number of rows for validation
    original_df_rows = df_categorical.shape[0]

    df_categorical = df_categorical.apply(round)
    for col, (min_val, max_val) in categorical_data_range.items():
        df_categorical[col] = df_categorical[col].apply(lambda x: max(min(x, max_val), 0))

    # Result validation
    assert df_categorical.shape[0] <= original_df_rows, "Dataframe should not be bigger after removing outliers"

    return df_categorical

# Imputes missing data in categorical and continuous columns separately
def impute_df(df,categorical_imputer, continuous_imputer,
              all_column_names,categorical_col_names,continuous_col_names,
              categorical_data_range):
    # reset index for df if the indexes are not in proper order
    df.reset_index(drop=True, inplace=True)

    # Provide a mask of missing data
    indicator = MissingIndicator(features="all")
    mask_missing_vals = indicator.fit_transform(df)
    
    transformer = ColumnTransformer(transformers=[
        ("categorical_imputer", categorical_imputer, categorical_col_names),
        ("continuous_imputer", continuous_imputer, continuous_col_names),
    ],remainder="passthrough")

    # perform imputation
    imputed_df = transformer.fit_transform(df)

    # split imputed data into categorical, continuous and target for further processing
    df_categorical = pd.DataFrame(imputed_df[:,:len(categorical_col_names)],columns=categorical_col_names)
    df_continuous = pd.DataFrame(imputed_df[:,len(categorical_col_names):len(all_column_names)-1],columns=continuous_col_names)
    df_target = df.iloc[:, -1]
    # ensure the categorical data are rounded to nearest values
    df_categorical = enforce_categorical_values(df_categorical, categorical_data_range)

    # assert number of columns are the same
    assert df_categorical.shape[1] == len(categorical_col_names), "Incorrect number of categorical columns after imputation"
    assert df_continuous.shape[1] == len(continuous_col_names), "Incorrect number of continuous columns after imputation"

    # combine df and return
    combined_imputed_df = pd.concat([df_categorical, df_continuous, df_target], axis=1)
    combined_imputed_df = combined_imputed_df[all_column_names]

    # Result validation
    assert df.shape[0] == combined_imputed_df.shape[0], "Dataframe should not be bigger after removing outliers"
    assert df.shape[1] == combined_imputed_df.shape[1], "Dataframe should not be bigger after removing outliers"

    return mask_missing_vals, transformer, combined_imputed_df


# This function will take a df, and the name of the target column and split it to x and y data
def split_data_to_x_y(df, target_column="TenYearCHD"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

# Performs SMOTE oversampling to balance class distribution
def smote_over_sampling(df, categorical_features, sampling_strategy="auto", seed=42, k_neighbors=5):
    # get the oversample
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=seed, sampling_strategy="minority", k_neighbors=5)
    X, y = split_data_to_x_y(df) 
    oversampled_features, oversampled_target = smote_nc.fit_resample(X, y)

    # Create a new DataFrame with the oversampled data and target variable
    oversampled_df = pd.DataFrame(oversampled_features, columns=X.columns)
    oversampled_df['TenYearCHD'] = oversampled_target

    # reset index for df if the indexes are not in proper order
    oversampled_df.reset_index(drop=True, inplace=True)

    # Result validation
    assert df.shape[1] == oversampled_df.shape[1], "Dataframe should not be bigger after removing outliers"

    return oversampled_df

# Returns a ColumnTransformer object to scale continuous columns using the StandardScaler
def custom_standard_scaler(df_features, continuous_col_names):
    # Use standard scaler and only scale the df with continuous values
    scaler = StandardScaler()
    ct_scaler = ColumnTransformer(
        [("preprocess_continuous", scaler, continuous_col_names)],
        remainder="passthrough"
    )

    # Fit the ColumnTransformer with the continuous data
    ct_scaler.fit(df_features[continuous_col_names])
    
    return ct_scaler


# Scales the input DataFrame using the Min-Max Scaler and returns the scaled DataFrame
def min_max_scaler(df_features):
    # create scaler and fit data
    scaler = MinMaxScaler()
    scaler.fit(df_features)

    # Transform the DataFrame with the scaled values
    df_scaled = pd.DataFrame(scaler.transform(df_features), columns=df_features.columns)

    return df_scaled, scaler

# Scales continuous columns of the input DataFrame using the custom scales
def scale_data_using_custom_scaler(df, ct_scaler, categorical_col_names, continuous_col_names):
    # save original df column names and split df to categorical, continuous, target
    data_col_names = df.columns
    df_continuous = df.loc[:, continuous_col_names]
    df_categorical = df.loc[:, categorical_col_names]

    # transform the data
    df_continuous_transformed = ct_scaler.transform(df_continuous)
    transformed_df = pd.DataFrame(df_continuous_transformed, columns=continuous_col_names)
    transformed_df = pd.concat([transformed_df, df_categorical], axis=1)

    # Rearrange columns to match the original order
    transformed_df = transformed_df[data_col_names]

    # Result validation
    assert df.shape[0] == transformed_df.shape[0], "Dataframe should not be bigger after removing outliers"
    assert df.shape[1] == transformed_df.shape[1], "Dataframe should not be bigger after removing outliers"

    return transformed_df

"""
#########################################################################################################################################################
#########################################################################################################################################################

This section contains the code used for feature selection

#########################################################################################################################################################
#########################################################################################################################################################
"""

'''
From Workshop
Create a pair-wise correlation plot.
'''
def make_pairplot(df, hue_target):
  sns.set(style='darkgrid')
  sns.pairplot(df, hue=hue_target, height=1, aspect=1.6, palette='tab10')
  plt.show()


'''
From Workshop
Generate a Heat Map from a correlational matrix and
save it to a file.
'''
def make_heatmap(corr_mat):
    plt.figure(figsize=(12,10))
    sns.heatmap(data=corr_mat,     # correlation matrix
    annot=True,             # display pearson correlation values
    annot_kws={'size':8},   # font size for values
    cmap='GnBu')
    plt.show()


'''
From workshop
Performs feature-selection. It selects the best features by looking
for features that correlates strongly w.r.t. the label. It also
removes redunduncy by eliminating peers that are strongly-correlated.
'''
def best_features(corr_mat, label, label_limit, peers_limit):

    # only consider features that are highly-correlated with our label
    candidates_df = corr_mat.loc[
    (corr_mat[label] < -label_limit) | (corr_mat[label] > label_limit),   # index
    (corr_mat[label] < -label_limit) | (corr_mat[label] > label_limit)    # column
    ] 

    # move our 'label' column to the end for easier processing later
    candidates_df = candidates_df.drop(columns=[label]) # remove 'label' from the columns
    candidates_df[label] = corr_mat[label] # move 'label' column to the last column

    # use this to store the best features so far
    accept = [] 

    # iteratively compares features against peers and the label
    while len(candidates_df.columns) > 1:  # stop when left with only our label
        # inspect each feature in turn
        feature = candidates_df.columns[0]

        # get all peers of these feature, except the label
        peers = candidates_df.loc[feature].drop(label)
        print("peers of '", feature, "' =\n", peers, '\n', sep='')

        # look for other features that are highly-correlated with 
        # the current 'feature'. only consider positively correlated 
        # to remove redundancy.
        high_corr = peers[peers > peers_limit]   
        print('high_corr =\n', high_corr, '\n', sep='')

        # extract the pearson correlation values of each of these 
        # highly-correlated features w.r.t. our label
        alike = candidates_df.loc[high_corr.index, label]
        print('alike =\n', alike, '\n', sep='')

        # idxmax() to get the feature that is most correlated with 
        # our label. abs() to absolute the values because 
        # the features could be either positively or negatively 
        # correlated to our label
        top = alike.abs().idxmax()  # row-label (feature-name) of max-value
        accept.append(top)

        # place index-names into an array, so that we can use them
        # as parameters to the drop() function
        alike = list(alike.index) 

        # done with feature, remove feature from 'candidates_df',
        # this allows our candidates to be smaller each time
        candidates_df = candidates_df.drop(columns=alike, index=alike)

    # returns best features in the    
    return accept


# Univariate Feature Selection using anova f value
def select_kbest_f_value(df, scaler, num_of_features, categorical_col_names, continuous_col_names):
    # assertions for input
    assert not df.empty, "Input DataFrame 'df' is empty."
    assert num_of_features > 0, "Parameter 'num_of_features' must be a positive integer."

    # Split data into X and y, return scaler object and get column names
    X, y, feature_names = split_scale_data_return_column_headers(df, scaler, categorical_col_names, continuous_col_names)

    # Using ANOVA F-value metric
    selector = SelectKBest(f_classif, k=num_of_features)
    selector.fit_transform(X, y)

    # Get the boolean mask of selected features
    mask = selector.get_support()

    # Get the feature names of selected features
    selected_features = X.columns[mask]

    # Return the selected feature names
    return list(selected_features)

# Splits and scales data, returning feature names
def split_scale_data_return_column_headers(df, scaler, categorical_col_names, continuous_col_names):
    # assertions for input
    assert not df.empty, "Input DataFrame 'df' is empty."

    # Split data into X and y
    X, y = split_data_to_x_y(df)

    # Fit the data to the scaler
    X = scale_data_using_custom_scaler(X, scaler, categorical_col_names, continuous_col_names)
    feature_names = X.iloc[:, :-1].columns

    return X, y, feature_names

# Prints RFECV results and returns selected features
def print_results_and_get_features(rfecv, feature_names, num_of_features):
    # assertions for input
    assert isinstance(rfecv, RFECV), "Input 'rfecv' is not a valid RFECV object."
    assert all(isinstance(feature, str) for feature in feature_names), "Invalid 'feature_names' list."
    assert num_of_features > 0, "Parameter 'num_of_features' must be a positive integer."

    # Create a DataFrame with the ranking and feature names
    df_ranking = pd.DataFrame(zip(rfecv.ranking_, feature_names), columns=['Ranking', 'Feature'])

    # Sort the DataFrame based on the ranking (in ascending order)
    df_ranking_sorted = df_ranking.sort_values(by='Ranking')

    # Get the topfeatures as a list
    selected_features = df_ranking_sorted.head(num_of_features)['Feature'].tolist()

    # Display the sorted DataFrame
    print(df_ranking_sorted)

    # Display the top features
    print("Top Features:")
    print(selected_features)

    return selected_features


# Wraps feature selection using RFECV and returns selected features
def feat_select_rfecv_using_model(model, df, scaler, n_splits, num_of_features
                    , categorical_col_names, continuous_col_names
                    , scoring="recall", seed=42):
    # assertions for input
    assert not df.empty, "Input DataFrame 'df' is empty."
    assert n_splits > 0, "Parameter 'n_splits' must be a positive integer."
    assert num_of_features > 0, "Parameter 'num_of_features' must be a positive integer."

    # Split data into X and y, and get column names
    X, y, feature_names = split_scale_data_return_column_headers(df, scaler, categorical_col_names, continuous_col_names)

    rfecv = RFECV(model, cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                  , scoring=scoring, n_jobs=-1)
    rfecv.fit(X, y)

    selected_features = print_results_and_get_features(rfecv, feature_names, num_of_features)
    return selected_features

# Saves feature selection and model evaluation results to a Pandas Series when using RFE
def save_model_results_to_series(method_name, curr_num_of_features
            ,features, results_df, model_list, train_df, n_splits, seed, show=True):
    # assertions for input
    assert method_name, "Method name cannot be empty."
    assert isinstance(curr_num_of_features, int) and curr_num_of_features > 0, "Parameter 'curr_num_of_features' must be a positive integer."
    assert all(isinstance(feature, str) for feature in features), "Invalid 'features' list."
    assert not train_df.empty, "Input DataFrame 'train_df' is empty."
    assert isinstance(n_splits, int) and n_splits > 0, "Parameter 'n_splits' must be a positive integer."
    assert isinstance(seed, int) and seed >= 0, "Parameter 'seed' must be a non-negative integer."
    assert isinstance(show, bool), "Parameter 'show' must be a boolean value."

    # Initialize an empty Series with the same column index as df_save
    temp_series = pd.Series(None, index=results_df.columns, dtype=object)
    temp_series.at["method_for_feature_selection_using_RFE"] = method_name
    temp_series.at["number_of_features_chosen"] = curr_num_of_features
    temp_series.at["selected_features_(list)"] = features

    # Perform feature selection
    df_features = train_df[features].copy()

    # Add the target column to the df_selected DataFrame using .loc
    df_features["TenYearCHD"] = train_df["TenYearCHD"]

    train_df = df_features.copy()

    # Fill up the DataFrame with model names and metrics
    for idx, (model, model_name) in enumerate(model_list):
        model_idx = idx + 1
        col_prefix = f"{model_idx}_"

        # Add the model name
        temp_series.at[f"Machine_Learning_Model_{model_idx}"] = model_name
        
        # Add the metrics for the model
        results_list = return_model_results(train_df, model, n_splits, seed, show)

        temp_series.at[f"{col_prefix}Average_Accuracy"] = results_list["average_accuracy"]
        temp_series.at[f"{col_prefix}Average_Recall"] = results_list["average_recall"]
        temp_series.at[f"{col_prefix}Average_Precision"] = results_list["average_precision"]
        temp_series.at[f"{col_prefix}Average_F1_score"] = results_list["average_f1_score"]

        # Reshape the 2-row matrix into a 1-row matrix
        confusion_matrix_reshaped = results_list["average_confusion_matrix"].flatten()
        confusion_matrix_to_string = ','.join(map(str, confusion_matrix_reshaped))

        temp_series.at[f"{col_prefix}confusion_matrix"] = confusion_matrix_to_string
        
    temp_series = temp_series.dropna()
    
    return temp_series

"""
#########################################################################################################################################################
#########################################################################################################################################################

This section contains the code used for feature extraction

#########################################################################################################################################################
#########################################################################################################################################################
"""

# Applies Principal Component Analysis (PCA) on the input data to reduce the number of dimensions to n_components
def apply_pca(n_components, data_np):
    pca = PCA(n_components=n_components)
    return pca, pca.fit_transform(X=data_np)

# Creates a new DataFrame from the given NumPy array data_np by dropping the specified label column and adding it back as the target column.
def make_dataframe_from(ref_data_df, data_np, label):
    data_df = pd.DataFrame(
        data=data_np, 
        columns=ref_data_df.columns.drop(label)
    )
    data_df[label] = ref_data_df[label]
    return data_df

# Determines the minimum number of components needed to explain at least percent of the total variance using PCA
def min_components(data_np, percent):
    min_components_needed = 1

    # Exclude the last column (label column) from the shape[1] count
    for min_components_needed in range(1, data_np.shape[1]):
        pca = PCA(n_components=min_components_needed)
        pca.fit(data_np[:, :-1])  # Exclude the last column (label column) when fitting PCA
        if pca.explained_variance_ratio_.sum() >= percent:
            return min_components_needed

# Performs feature extraction using PCA to reduce the number of dimensions based on the target
# explained variance and returns a new DataFrame with transformed features
def feat_extraction_pca(df_selected, target_explained_variance, target_column="TenYearCHD"):
    # Assume you have a pandas DataFrame named 'df_selected' with the features and a target column named 'TenYearCHD'
    # Check if target_explained_variance is within a valid range
    assert 0 < target_explained_variance < 1, "target_explained_variance must be between 0 and 1 (exclusive)."

    # Step 1: Convert DataFrame to NumPy array
    data_np = df_selected.drop(columns=['TenYearCHD']).to_numpy()
    target_column = "TenYearCHD"

    # Step 2: Determine the number of components to aim for based on explained variance
    target_explained_variance = target_explained_variance # You can set this to your desired threshold
    min_pc = min_components(data_np, target_explained_variance)

    # Step 3: Apply PCA with the determined number of components
    pca, new_features = apply_pca(min_pc, data_np)

    # Step 4: Create a new DataFrame with the transformed features and the target column
    selected_feature_names = [f'PC{i+1}' for i in range(min_pc)]  # Naming the new features as PC1, PC2, etc.
    df_pca = pd.DataFrame(data=new_features, columns=selected_feature_names)
    df_pca[target_column] = df_selected[target_column]

    # The 'pca_df' DataFrame now contains the transformed features obtained through PCA, along with the target column.

    # Explained variance ratios captured by each of the principal components
    print("Explained Variance Ratios =", pca.explained_variance_ratio_)

    # Total explained variance captured by all the principal components
    print("Total Explained Variance =", pca.explained_variance_ratio_.sum())

    # The number of principal components required to capture at least 85% of the information of the original dataset
    print("Min principal components =", min_pc)

    return df_pca


# Performs Kernel PCA with a linear kernel on the input DataFrame to extract new features 
# and returns a new DataFrame with the transformed features
def feat_extraction_kernel_pca(df, n_components, kernel="linear", target_column="TenYearCHD"):
    X, y = split_data_to_x_y(df)
    transformer = KernelPCA(n_components=n_components, kernel=kernel)
    X_transformed = transformer.fit_transform(X)

    # Create a new DataFrame from the new components
    selected_feature_names = [f'Kernel_PC{i+1}' for i in range(n_components)]  # Naming the new features as Kernel_PC1, PC2, etc.
    df_kernel_pca = pd.DataFrame(data=X_transformed, columns=selected_feature_names)
    df_kernel_pca[target_column] = y

    return df_kernel_pca

# Applies Projection Pursuit Analysis to the features of the input DataFrame and 
# returns a new DataFrame with the transformed features
def feat_extraction_project_pursuit(df, n_components, target_column="TenYearCHD"):
    X, y = split_data_to_x_y(df)

    # Apply Projection Pursuit Analysis to the features
    pp_regressor = ProjectionPursuitRegressor(r=n_components)
    transformed_features = pp_regressor.fit_transform(X, y)

    # Create a new DataFrame with the transformed features
    df_pp = pd.DataFrame(data=transformed_features, columns=[f'PP{i+1}' for i in range(transformed_features.shape[1])])
    df_pp[target_column] = y

    return df_pp

# Saves the results of feature extraction using different machine learning models to a Pandas Series
def save_feat_extract_results_to_series(method_name
            , results_df, model_list, train_df, n_splits, seed, show=True):
    # Initialize an empty Series with the same column index as df_save
    temp_series = pd.Series(None, index=results_df.columns, dtype=object)
    temp_series.at["method_for_feature_extraction"] = method_name
    temp_series.at["number_of_extracted_components"] = train_df.shape[1] - 1

    # Fill up the DataFrame with model names and metrics
    for idx, (model, model_name) in enumerate(model_list):
        model_idx = idx + 1
        col_prefix = f"{model_idx}_"

        # Add the model name
        temp_series.at[f"Machine_Learning_Model_{model_idx}"] = model_name
        
        # Add the metrics for the model
        results_list = return_model_results(train_df, model, n_splits, seed, show)

        temp_series.at[f"{col_prefix}Average_Accuracy"] = results_list["average_accuracy"]
        temp_series.at[f"{col_prefix}Average_Recall"] = results_list["average_recall"]
        temp_series.at[f"{col_prefix}Average_Precision"] = results_list["average_precision"]
        temp_series.at[f"{col_prefix}Average_F1_score"] = results_list["average_f1_score"]

        # Reshape the 2-row matrix into a 1-row matrix
        confusion_matrix_reshaped = results_list["average_confusion_matrix"].flatten()
        confusion_matrix_to_string = ','.join(map(str, confusion_matrix_reshaped))

        temp_series.at[f"{col_prefix}confusion_matrix"] = confusion_matrix_to_string
    
    return temp_series

"""
#########################################################################################################################################################
#########################################################################################################################################################

This section contains the code used for model selection and training

#########################################################################################################################################################
#########################################################################################################################################################
"""

# Display average confusion matrix and evaluation metrics
def display_results(average_confusion_matrix, average_accuracy,
                  average_recall, average_precision,average_f1_score):
    print(f"Average Confusion Matrix:\n{average_confusion_matrix}\n")
    print(f"Average Accuracy: {average_accuracy:.2f}")
    print(f"Average Recall: {average_recall:.2f}")
    print(f"Average Precision: {average_precision:.2f}")
    print(f"Average F1 Score: {average_f1_score:.2f}")

# Evaluate a machine learning model using k-fold cross-validation and display average confusion matrix and evaluation metrics
def evaluate_model(X, y, model, n_splits, seed, show=True):
    # Perform k-fold cross-validation and get predicted labels for each fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred_cv = cross_val_predict(model, X, y, cv=skf)

    confusion_matrices = []

    # Calculate and store the confusion matrix for each fold
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        y_true_fold = y[test_index]
        y_pred_fold = y_pred_cv[test_index]
        cm_fold = confusion_matrix(y_true_fold, y_pred_fold)
        confusion_matrices.append(cm_fold)

    # Calculate the average confusion matrix across all folds
    average_confusion_matrix = np.mean(confusion_matrices, axis=0)

    # Calculate metrics directly from the average confusion matrix
    tn, fp, fn, tp = average_confusion_matrix.ravel()

    # Calculate the average accuracy
    average_accuracy = (tp + tn) / (tp + fp + fn + tn)

    # Calculate the average recall
    average_recall = tp / (tp + fn)

    # Calculate the average precision
    average_precision = tp / (tp + fp)

    # Calculate the average F1 score
    average_f1_score = 2 * tp / (2 * tp + fp + fn)
    
    if show:
        display_results(average_confusion_matrix, average_accuracy,
                    average_recall, average_precision,average_f1_score)


# Evaluate a machine learning model using k-fold cross-validation and return evaluation metrics as a dictionary
def return_model_results(df, model, n_splits, seed, show):
    # assertions for input
    assert isinstance(df, pd.DataFrame), "df_save must be a Pandas DataFrame."

    X, y = split_data_to_x_y(df)

    # Perform k-fold cross-validation and get predicted labels for each fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred_cv = cross_val_predict(model, X, y, cv=skf)

    confusion_matrices = []

    # Calculate and store the confusion matrix for each fold
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        y_true_fold = y[test_index]
        y_pred_fold = y_pred_cv[test_index]
        cm_fold = confusion_matrix(y_true_fold, y_pred_fold)
        confusion_matrices.append(cm_fold)

    # Calculate the average confusion matrix across all folds
    average_confusion_matrix = np.mean(confusion_matrices, axis=0)

    # Calculate metrics directly from the average confusion matrix
    tn, fp, fn, tp = average_confusion_matrix.ravel()

    # Calculate the average accuracy
    average_accuracy = (tp + tn) / (tp + fp + fn + tn)

    # Calculate the average recall
    average_recall = tp / (tp + fn)

    # Calculate the average precision
    average_precision = tp / (tp + fp)

    # Calculate the average F1 score
    average_f1_score = 2 * tp / (2 * tp + fp + fn)
    
    if show:
        display_results(average_confusion_matrix, average_accuracy,
                    average_recall, average_precision,average_f1_score)


    results_list = {"average_accuracy": average_accuracy,
                    "average_recall": average_recall,
                    "average_precision": average_precision,
                    "average_f1_score": average_f1_score,
                    "average_confusion_matrix": average_confusion_matrix}

    return results_list


# From Chatgpt
# Plot a confusion matrix heatmap with annotations
def plot_confusion_matrix(confusion_matrix, title, ax):
    sns.heatmap(confusion_matrix, annot=True, fmt='.1f', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

# Visualize the results of machine learning model evaluation with confusion matrices and evaluation metrics
def visualize_results(df_save):
    # assertions for input
    assert isinstance(df_save, pd.DataFrame), "df_save must be a Pandas DataFrame."

    num_matrices_per_row = 5
    num_rows = len(df_save)
    figure_height = max(15, num_rows * 2.5)  # Set a minimum height of 15 inches

    fig, axes = plt.subplots(num_rows, num_matrices_per_row, figsize=(20, figure_height),
                             gridspec_kw={'width_ratios': [0.5, 3, 3, 3, 3]})

    for i, row in df_save.iterrows():
        for j in range(num_matrices_per_row):
            if j == 0:
                num_selected_features = row['number_of_features_chosen']  # Replace 'number_of_selected_features' with the correct column name

                info_text = f"Number of \nselected features: {num_selected_features}"
                axes[i, 0].text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12)
                axes[i, 0].axis('off')

            elif j == 1:
                # Column 2: Metrics
                data = {
                    row["Machine_Learning_Model_1"]: [row["1_Average_Accuracy"], row["1_Average_Recall"],
                                                      row["1_Average_Precision"], row["1_Average_Accuracy"]],
                    row["Machine_Learning_Model_2"]: [row["2_Average_Accuracy"], row["2_Average_Recall"],
                                                      row["2_Average_Precision"], row["2_Average_F1_score"]],
                    row["Machine_Learning_Model_3"]: [row["3_Average_Accuracy"], row["3_Average_Recall"],
                                                      row["3_Average_Precision"], row["3_Average_F1_score"]]
                }
                index = ['Average accuracy', 'Average recall', 'Average precision', 'Average f1 score']
                blank_df = pd.DataFrame(data, index=index)

                # Ensure the blank DataFrame contains only NaN (float) values
                blank_df = blank_df.astype(float)

                # Plot the DataFrame table in column 1, row 2
                sns.heatmap(blank_df, annot=True, fmt='.3f', cmap='Blues', cbar=False, ax=axes[i, 1], vmin=0, vmax=1,
                            annot_kws={"size": 12})  # Adjust the font size for the DataFrame cell values

                axes[i, 1].set_title('Metrics', fontsize=14)
                axes[i, 1].set_xticklabels(axes[i, 1].get_xticklabels(), rotation=45, ha='right', fontsize=10)

                # Explicitly set the x and y-axis limits to avoid UserWarning
                axes[i, 1].set_xlim(0, len(blank_df.columns))
                axes[i, 1].set_ylim(0, len(blank_df.index))

            else:
                # Adjust indices here based on your actual data
                model_name = ""
                if j == 2:
                    model_name = row["Machine_Learning_Model_1"]
                elif j == 3:
                    model_name = row["Machine_Learning_Model_2"]
                elif j == 4:
                    model_name = row["Machine_Learning_Model_3"]

                # Similarly, adjust the column indices here based on your actual data
                confusion_matrix_data = list(map(float, row[f'{j-1}_confusion_matrix'].split(',')))
                confusion_matrix = pd.DataFrame([confusion_matrix_data[:2], confusion_matrix_data[2:]],
                                                index=['True Negative', 'True Positive'],
                                                columns=['Predicted Negative', 'Predicted Positive'])

                # Plot the confusion matrix with adjusted cell size
                plot_confusion_matrix(confusion_matrix, title=f"Confusion Matrix ({model_name})", ax=axes[i, j])  # Adjust the colorbar size for the heatmap
                axes[i, j].tick_params(axis='both', which='both', labelsize=7)  # Adjust the font size for tick labels

    plt.tight_layout()
    plt.show()

# Visualize the results of feature extraction with machine learning models using confusion matrices and evaluation metrics
def visualize_feat_extract_results(df_save):
    # assertions for input
    assert isinstance(df_save, pd.DataFrame), "df_save must be a Pandas DataFrame."

    num_matrices_per_row = 5
    num_rows = len(df_save)
    figure_height = max(15, num_rows * 2.5)  # Set a minimum height of 15 inches

    fig, axes = plt.subplots(num_rows, num_matrices_per_row, figsize=(20, figure_height),
                             gridspec_kw={'width_ratios': [0.5, 3, 3, 3, 3]})

    for i, row in df_save.iterrows():
        for j in range(num_matrices_per_row):
            if j == 0:
                method_name = row['method_for_feature_extraction']  # Replace 'number_of_selected_features' with the correct column name
                num_components = row['number_of_extracted_components']  # Replace 'number_of_selected_features' with the correct column name
                info_text = f"Method: {method_name}\n\nNumber of extracted components: {num_components}"
                axes[i, 0].text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12)
                axes[i, 0].axis('off')

            elif j == 1:
                # Column 2: Metrics
                data = {
                    row["Machine_Learning_Model_1"]: [row["1_Average_Accuracy"], row["1_Average_Recall"],
                                                      row["1_Average_Precision"], row["1_Average_Accuracy"]],
                    row["Machine_Learning_Model_2"]: [row["2_Average_Accuracy"], row["2_Average_Recall"],
                                                      row["2_Average_Precision"], row["2_Average_F1_score"]],
                    row["Machine_Learning_Model_3"]: [row["3_Average_Accuracy"], row["3_Average_Recall"],
                                                      row["3_Average_Precision"], row["3_Average_F1_score"]]
                }
                index = ['Average accuracy', 'Average recall', 'Average precision', 'Average f1 score']
                blank_df = pd.DataFrame(data, index=index)

                # Ensure the blank DataFrame contains only NaN (float) values
                blank_df = blank_df.astype(float)

                # Plot the DataFrame table in column 1, row 2
                sns.heatmap(blank_df, annot=True, fmt='.3f', cmap='Blues', cbar=False, ax=axes[i, 1], vmin=0, vmax=1,
                            annot_kws={"size": 12})  # Adjust the font size for the DataFrame cell values

                axes[i, 1].set_title('Metrics', fontsize=14)
                axes[i, 1].set_xticklabels(axes[i, 1].get_xticklabels(), rotation=45, ha='right', fontsize=10)

                # Explicitly set the x and y-axis limits to avoid UserWarning
                axes[i, 1].set_xlim(0, len(blank_df.columns))
                axes[i, 1].set_ylim(0, len(blank_df.index))

            else:
                # Adjust indices here based on your actual data
                model_name = ""
                if j == 2:
                    model_name = row["Machine_Learning_Model_1"]
                elif j == 3:
                    model_name = row["Machine_Learning_Model_2"]
                elif j == 4:
                    model_name = row["Machine_Learning_Model_3"]

                # Similarly, adjust the column indices here based on your actual data
                confusion_matrix_data = list(map(float, row[f'{j-1}_confusion_matrix'].split(',')))
                confusion_matrix = pd.DataFrame([confusion_matrix_data[:2], confusion_matrix_data[2:]],
                                                index=['True Negative', 'True Positive'],
                                                columns=['Predicted Negative', 'Predicted Positive'])

                # Plot the confusion matrix with adjusted cell size
                plot_confusion_matrix(confusion_matrix, title=f"Confusion Matrix ({model_name})", ax=axes[i, j])  # Adjust the colorbar size for the heatmap
                axes[i, j].tick_params(axis='both', which='both', labelsize=7)  # Adjust the font size for tick labels

    plt.tight_layout()
    plt.show()

# Save the results of hyperparameter tuning in a specific column of the DataFrame for the corresponding machine learning model
def save_tuning_results_in_col(df, ml_model_name, column_name, results):
    # assertions for input
    assert isinstance(df, pd.DataFrame), "df must be a Pandas DataFrame."
    assert isinstance(ml_model_name, str), "ml_model_name must be a string."
    assert isinstance(column_name, str), "column_name must be a string."

    idx = df.loc[df['ML_Model'] == ml_model_name].index[0]
    if not isinstance(results, float):
        results = str(results)
    df.loc[idx, column_name] = str(results)  # Convert random_grid to a string before storing
    return df