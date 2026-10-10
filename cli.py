import numpy as np
import pandas as pd
import mysql.connector
from getpass import getpass
from pipeline import lifecycle

def start(data_type=None, start_params=None, act=None):

    alpha = 0.00004
    omega = 0.0007
    iterations = 9000

    if act is None:
        act = 1
    elif act == 0:
        act = 0
    else:
        print("You entered an invalid action!")

    if data_type == "csv":
        if start_params is None:
            start_params = pd.read_csv("Parameters.csv")
        else:
            start_params = pd.read_csv(start_params)
        file_name = str(start_params.iloc[0, 0])
        column_name = str(start_params.iloc[2, 0])
        model_type = str(start_params.iloc[4, 0]).strip()
        columns_name = start_params.iloc[6].dropna().astype(str).str.strip().tolist()
        test_list = start_params.iloc[8].dropna().tolist()

    elif data_type == "database":
        server_params = input("Enter your server parameters separated by spaces (host port user): ")
        server_params = server_params.split(" ")
        password = getpass("Enter your server password: ")
        database = input("Enter your database name: ")
        table = input("Enter the table you want to use for training and testing: ")

        # MySQL connection
        connection = mysql.connector.connect(host=server_params[0], port=int(server_params[1]), user=server_params[2], password=password, database=database)
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(f"select * from `{table}`;")
                df = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
            finally:
                cursor.close()
        finally:
            connection.close()

        # Preparing data
        file_name = df
        data_for_preprocessing = input("Enter the target column and data type (class or property), separated by a space: ")
        data_for_preprocessing = data_for_preprocessing.split(" ")
        column_name = data_for_preprocessing[0]
        model_type = data_for_preprocessing[1]
        columns_name = input("Enter the columns you do not need, separated by spaces: ")
        columns_name = columns_name.split(" ")
        test_list = input("Enter the features of the object to predict, separated by spaces: ")
        test_list = test_list.split(" ")

    else:
        print("Specify the data source in 'start' (csv or database)")
        return

    if model_type == "class":
        model_type = "classification"
    elif model_type == "property":
        model_type = "linear_regression"
    else:
        print("You entered an invalid prediction type!")

    # Prediction
    prediction = lifecycle(file_name=file_name, column_name=column_name, model_type=model_type, act=act, columns_name=columns_name, alpha=alpha, omega=omega, iterations=iterations, test_list=test_list)
    if prediction[0] == "classification:":

        probability = float(prediction[1][0])
        predicted_class = int(probability >= 0.5)

        print(predicted_class)

    elif prediction[0] == "linear_regression:":
        w = np.array(prediction[1])
        b = prediction[2]
        z = prediction[-2] @ w + b
        params = prediction[-1]
        z = z * params["y_std"] + params["y_mean"]
        print(z)

    elif prediction[0] in ["decision_tree:", "random_forest:"]:
        print(float(np.mean(prediction[1])))
    else:
        print(prediction)
