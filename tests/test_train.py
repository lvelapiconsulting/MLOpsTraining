from model.train import get_csvs_df
import os
import pytest
from model.train import split_data
import pandas as pd
from model.train import train_model
import numpy as np

def test_csvs_no_files():
    with pytest.raises(RuntimeError) as error:
        get_csvs_df("./")
    assert error.match("No CSV files found in provided data")


def test_csvs_no_files_invalid_path():
    with pytest.raises(RuntimeError) as error:
        get_csvs_df("/invalid/path/does/not/exist/")
    assert error.match("Cannot use non-existent path provided")


def test_csvs_creates_dataframe():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    datasets_directory = os.path.join(current_directory, 'datasets')
    result = get_csvs_df(datasets_directory)
    assert len(result) == 20
    def test_split_data_missing_features():
        df = pd.DataFrame({'col1': [1, 2, 3]})
        with pytest.raises(ValueError) as error:
            split_data(df)
        assert error.match("The dataframe does not contains the features")


def test_split_data_valid():
    data = {
        'Pregnancies': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'PlasmaGlucose': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195],
        'DiastolicBloodPressure': [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
        'TricepsThickness': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        'SerumInsulin': [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        'BMI': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
        'DiabetesPedigree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        'Age': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        'Diabetic': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) == 14
    assert len(X_test) == 6
    assert len(y_train) == 14
    assert len(y_test) == 6

def test_train_model_executes_successfully():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    X_test = np.array([[2, 3], [6, 7]])
    y_train = np.array([0, 1, 0, 1])
    y_test = np.array([1, 0])
    train_model(0.01, X_train, X_test, y_train, y_test)

def test_train_model_with_different_reg_rate():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    X_test = np.array([[2, 3], [6, 7]])
    y_train = np.array([0, 1, 0, 1])
    y_test = np.array([1, 0])
    train_model(0.1, X_train, X_test, y_train, y_test)
    train_model(0.5, X_train, X_test, y_train, y_test)


