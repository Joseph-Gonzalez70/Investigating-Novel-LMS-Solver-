# This file contains functions that remove unnecessary fields
# from the Car_Purchasing_Data.csv, Austin_Housing_Data.csv, and electric motor temperature data sets
# The import_adjust_electric_data() function also imports the electric motor temperature data
# To get more information about the variables, see data summary jupyter notebooks

# Libraries:
import pandas as pd
from os import listdir
from re import search

def adjust_housing_data(house_data):
    """
    This function removes the  following columns from the housing data:
          'zpid', 'zipcode', 'latest_salemonth', 'numOfPhotos'
    This function also transforms the yearBuilt and latest_saleyear columns.
    Input:
        House data as a dataframe.
    Output:
    X -> Observations for the predictor variables(numOfBathrooms, numOfBedrooms, etc.)
    y -> Response variable(latestPrice)
    """
    # Identify numeric columns and remove columns with string and character types(non-categorical)
    remove_columns = (house_data.dtypes == "float64") | (house_data.dtypes == "int64") | (house_data.dtypes == "bool")
    house_data = house_data.iloc[:, remove_columns.values]
    remove_columns = ~house_data.columns.isin(['zpid', 'zipcode', 'latest_salemonth', 'numOfPhotos'])
    house_data = house_data.iloc[:, remove_columns]
    # Change the yearBuilt and latest_saleyear columns:
    house_data.loc[:, "yearBuilt"] = 2021 - house_data.loc[:, "yearBuilt"]
    house_data.loc[:, "latest_saleyear"] = 2021 - house_data.loc[:, "latest_saleyear"]
    # Change the column names:
    house_data = house_data.rename(columns={'yearBuilt': 'houseAge', 'latest_saleyear': 'numYearsLastSale'})
    X = house_data.loc[:, house_data.columns != "latestPrice"]
    y = house_data.loc[:, "latestPrice"]
    return(X, y)


def adjust_car_data(car_data):
    """
     This function removes the  following columns from the car purchasing data:
              'Customer Name', 'Customer e-mail', 'Country'
    This function also transforms the 'Gender" variable to a boolean type.
     Input:
        Car purchasing data as a dataframe.
    Output:
    X -> Observations for the predictor variables(Gender, Age, Net Worth, etc.)
    y -> Response variable(Car Purchase Amount)
    """
    # Remove the object data type columns:
    car_data = car_data.iloc[:, 3:]
    # Make Gender a boolean:
    car_data.loc[:, "Gender"] = car_data.Gender.astype("bool")
    # Construct X and y:
    X = car_data.iloc[:, car_data.columns != "Car Purchase Amount"]
    y = car_data.loc[:, "Car Purchase Amount"]
    return(X, y)

def import_adjust_electric_data():
    """
    This function imports the data from each electric motor temperature data file.
    This function also removes the "profile_id" field.
    Output:
    X -> Observations for the predictor variables(motor_speed, i_d, i_q , etc.)
    y -> Response variable(pm)
    """
    files = listdir("../data")
    files.sort()
    motor_files = [i for i in files if bool(search("Electric", i))]
    # Import the data:
    electric_motor_data = []
    for i in range(0, len(motor_files)):
        path = "../data/" + motor_files[i]
        if i == 0:
            electric_motor_data.append(pd.read_csv(path, header=0))
            col_names = electric_motor_data[0].columns
            continue
        electric_motor_data.append(pd.read_csv(path, header=None, names=col_names))
    # Combine the data:
    e_m_data = pd.concat(electric_motor_data)
    # Remove the profile_id column
    e_m_data = e_m_data.iloc[:, e_m_data.columns != "profile_id"]
    # Get X and y:
    X = e_m_data.iloc[:,  e_m_data.columns != "pm"]
    y = e_m_data.iloc[:,  e_m_data.columns == "pm"]
    return(X, y)