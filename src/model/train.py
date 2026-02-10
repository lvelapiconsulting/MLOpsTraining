# Import libraries

import argparse
import glob
import os

import pandas as pd

import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV


# define functions
def main(args):
    print("Training model with the following parameters:")
    # TO DO: enable autologging
    mlflow.autolog()
    with mlflow.start_run():

        # read data
        df = get_csvs_df(args.training_data)

        # split data
        X_train, X_test, y_train, y_test = split_data(df)

        # train model
        train_model(args.reg_rate, X_train, X_test, y_train, y_test)

        # train random forest model
        random_forest_model(X_train, X_test, y_train, y_test)

        # hyperparameter tuning for random forest model
        hyperparameter_tuning_RF(X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    # evaluate the model on the test set
    test_score = model.score(X_test, y_test)
    print(f"Logistic Regression model test score: {test_score}")


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def random_forest_model(X_train, X_test, y_train, y_test):
    # train a random forest model
    model = RandomForestClassifier(n_estimators=10, max_depth=5,
                                   random_state=0)
    model.fit(X_train, y_train)
    # evaluate the model on the test set
    test_score = model.score(X_test, y_test)
    print(f"Random Forest model test score: {test_score}")


def hyperparameter_tuning_RF(X_train, X_test, y_train, y_test):
    # Create a random search space for the hyperparameters
    param_dist = {
        'n_estimators': [10, 50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # Perform random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=0),
        param_distributions=param_dist,
        n_iter=10,
        cv=2,
        random_state=0
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    # Evaluate the best model on the test set
    test_score = best_model.score(X_test, y_test)
    print(f"Best Random Forest model test score: {test_score}")
    print(f"Best RF hyperparameters: {random_search.best_params_}")


def split_data(df):
    # split in training and testing data
    features = ['Pregnancies', 'PlasmaGlucose',
                'DiastolicBloodPressure', 'TricepsThickness',
                'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']
    target = 'Diabetic'
    missed_features = [col for col in features + [target]
                       if col not in df.columns]
    if missed_features:
        msg = (f"DF features missed:\n{missed_features}")
        raise ValueError(msg)
    X, y = df[features].values, df[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
        )
    return X_train, X_test, y_train, y_test


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
