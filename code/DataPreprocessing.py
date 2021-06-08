# Data Import and Preprocessing
# This file preprocesses imports and preprocesses the data sets.
# Each function removes columns that are not useful for regression.
# The std_data parameter is an option for the data to be standardized
# The train_test_split parameter is an option for the set to be split
# into a test and training set.

# Note: For a further explanation on the steps(removing certain columns etc.) in each function,
# see the "Data Summary" Jupyter notebooks in the notebooks folder.

# Libraries:
import pandas as pd
import StandardizeSplit as sp
import AdjustData as ad


def process_data(X, y, std_data, train_test_split, test_split = 0.3):
    """
    This function splits and/or standardizes the data set.
    We standardize the data using the standardize_data function from StandardizeSplit.
    We split the data using the split_dataset function from StandardizeSplit.
    Input:
        X: Data corresponding to the predictor variables.(Array)
        y: Data corresponding to the response variable.
        std_data: Set to True to standardize the data.
        train_test_split: Set to True to get a training and test split
        test_split: The percentage of the full data to include in the test set.
    """
    if std_data == True and train_test_split == True:
        X_tr, y_tr, X_test, y_test = sp.split_dataset(X, y, test_split)
        X_tr, X_test = sp.standardize_data(X_tr, X_test)
        return (X_tr, y_tr, X_test, y_test)
    elif train_test_split == True:
        X_tr, y_tr, X_test, y_test = sp.split_dataset(X, y, test_split)
        return (X_tr, y_tr, X_test, y_test)
    elif std_data == True:
        X_tr = sp.standardize_data(X)
        return (X_tr, y)

def get_housing_data(std_data = False, train_test_split = False, test_split = 0.3):
    """
    This function imports Austin_Housing_Data.csv.
    Uses adjust_housing_data function from AdjustData.py to remove unnecessary columns.
    Output: depends on the std_data and train_test_split(default for test split 0.30)
    Parameters 1: std_data = False, train_test_split = False -> Output 1: Matrix X(nxP) and response y
    Parameters 2: std_data = True, train_test_split = False  -> Output 2: Standardized Matrix X and response y
    Parameters 3: std_data = False, train_test_split = True  -> Output 3: X_train, y_train, X_test, y_test
    Parameters 4: std_data = True, train_test_split = True   -> Output 4: Standardized X_train, y_train, Standardized X_test, y_test
    """
    # Import the data:
    path = "../data/Austin_Housing_Data.csv"
    house_data = pd.read_csv(path, header=0)

    # Get X and y:
    X, y = ad.adjust_housing_data(house_data)

    # Identify the options:
    if std_data == False and train_test_split == False:
        return(X, y)
    return(process_data(X,y,std_data, train_test_split, test_split))


def get_car_data(std_data = False, train_test_split = False, test_split = 0.3):
    """
    This function imports Car_Purchasing_Data.csv.
    Uses the adjust_car_data function from AdjustData.py to remove unnecessary columns.
    Output:
    Parameters 1: std_data = False, train_test_split = False -> Output 1: Matrix X(nxP) and response y
    Parameters 2: std_data = True, train_test_split = False  -> Output 2: Standardized Matrix X and response y
    Parameters 3: std_data = False, train_test_split = True  -> Output 3: X_train, y_train, X_test, y_test
    Parameters 4: std_data = True, train_test_split = True   -> Output 4: Standardized X_train and X_test, y_train, y_test
    """
    # Import the data set:
    path = "../data/Car_Purchasing_Data.csv.xls"
    car_data = pd.read_csv(path)

    # Get X and y:
    X, y = ad.adjust_car_data(car_data)
    
    # Identify the options:
    if std_data == False and train_test_split == False:
        return(X, y)
    return(process_data(X,y,std_data, train_test_split, test_split))
    

def get_motor_data(train_test_split = False, test_split = 0.3):
    """
     This function imports the electric motor temperature data(Normalized data).
     Uses the import_adjust_electric_data function from AdjustData.py to import data and remove unnecessary columns.
     Output:
     Parameters 1: train_test_split = False -> Output 1: Matrix X(nxP) and response y
     Parameters 2: train_test_split = True  -> Output 2: X_train, y_train, X_test, y_test
     """
    X, y = ad.import_adjust_electric_data()
    # Identify the options:
    std_data = False # The data is normalized
    if train_test_split == False:
        return(X, y)
    return(process_data(X, y, std_data, train_test_split, test_split))
