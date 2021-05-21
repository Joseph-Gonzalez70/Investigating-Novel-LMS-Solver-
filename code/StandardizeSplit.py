# This file contains functions for standardizing and splitting the data

# Libraries:
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def standardize_data(X_tr, X_test = None):
    """
    This function takes in at least a training set and standardizes it.
    If a test set is present, the test set is also standardized.
    Output:
    Parameters 1: X_tr, X_test -> Output 1: Standardized X_tr and X_test
    Parameters 2: X_tr, X_test = None -> Output 1: Standardized X_tr
    """
    std_x = StandardScaler()
    if X_test:
       X_tr = std_x.fit_transform(X_tr)
       return(X_tr)
    X_tr = std_x.fit_transform(X_tr)
    X_test = std_x.transform(X_tr)
    return(X_tr. X_test)

def split_dataset(X, y, test_split):
    """
    This function takes X data, y data and a test split proportion.
    The test split determines how much data is in the test X and test y.
    Output:
                X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_split)
    return(X_train, X_test, y_train, y_test )