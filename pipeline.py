import numpy as np
import pandas as pd
from models import (knn, linear_regression, classification, decision_tree, random_forest)
from preprocessing import preprocessing
from imputation import imputer

def pipeline(column_, data, ml_model, act, alpha, omega, iterations, test_list=None, columns_name=None):
    id_columns = [] if columns_name is None else list(columns_name)
    data = data.drop(columns=id_columns)
    scale_y = ml_model not in ["knn", "classification"]
    data = data.drop_duplicates().reset_index(drop=True)
    # data = data.drop_duplicates().iloc[:-1].reset_index(drop=True)
    folds = []
    for percent in range(0, 100, 20):
        start = int(len(data) * (percent * 0.01))
        end = int(len(data) * (percent * 0.01 + 0.2))
        fold = data.iloc[start:end].reset_index(drop=True)
        folds.append(fold)

    # Average regression metrics
    mse_avg = []
    rmse_avg = []
    r2_avg = []

    # Classification metrics
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Cross-validation
    predictions_for_answer = []
    for index, validation_fold in enumerate(folds):
        train_data = pd.concat([fold for i, fold in enumerate(folds) if i != index], ignore_index=True)
        validation_data = validation_fold.copy()

        train_y = train_data[column_].copy()
        validation_y = validation_data[column_].copy()
        train_features, imputer_models = imputer(file_name=train_data.drop(columns=column_), columns_name=None, alpha=alpha, iterations=iterations)
        validation_features, _ = imputer(file_name=validation_data.drop(columns=column_), columns_name=None, alpha=alpha, iterations=iterations, w_and_b=imputer_models)
        train_data = train_features.copy()
        train_data[column_] = train_y
        validation_data = validation_features.copy()
        validation_data[column_] = validation_y
        x_train, y_train, params = preprocessing(data=train_data, c=column_, scale_y=scale_y)
        x_test, y_test, _ = preprocessing(data=validation_data, c=column_, params=params, scale_y=scale_y)
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
        if act == 1:
            if test_list is not None:
                feature_columns = [col for col in train_data.columns if col != column_]
                test_df = pd.DataFrame([test_list], columns=feature_columns)

                test_df, _ = imputer(file_name=test_df, columns_name=None, alpha=alpha, iterations=iterations, w_and_b=imputer_models)
                test_df[column_] = train_data[column_].iloc[0]
                x_new, _, _ = preprocessing(data=test_df, c=column_, params=params, scale_y=scale_y)
                x_new = x_new.reindex(columns=x_train.columns, fill_value=0)
            else:
                x_new = x_test

        # Print
        if ml_model == "knn":
            k = min(5, len(x_train))
            if k % 2 == 0:
                k -= 1

            for i in range(len(x_test)):
                prediction = knn(x_train=x_train, y_train=y_train, x=x_test.iloc[i].values, k=k)
                y_true = np.asarray(y_test)[i]

                if prediction >= 0.5 and y_true >= 0.5:
                    tp += 1
                elif prediction < 0.5 and y_true < 0.5:
                    tn += 1
                elif prediction < 0.5 and y_true >= 0.5:
                    fn += 1
                else:
                    fp += 1

            if act == 1:
                new_prediction = knn(x_train=x_train, y_train=y_train, x=x_new.iloc[0].values, k=k)
                predictions_for_answer.append(new_prediction)

        elif ml_model == "linear_regression":
            w, b = linear_regression(x_train=x_train, y_train=y_train, alpha=alpha, iterations=iterations)
            prediction_scaled = x_test @ w + b
            prediction = prediction_scaled * params["y_std"] + params["y_mean"]
            y_test_real = y_test * params["y_std"] + params["y_mean"]
            if act == 1:
                predictions_for_answer.append(prediction)
            mse_ = (1 / len(y_test_real)) * ((prediction - y_test_real).T @ (prediction - y_test_real))
            mse_avg.append((mse_))
            rmse_avg.append(np.sqrt(mse_))
            r2_avg.append(1 - (((prediction - y_test_real).T @ (prediction - y_test_real)) / ((y_test_real - np.mean(y_test_real)).T @ (y_test_real - np.mean(y_test_real)))))

        elif ml_model == "classification":
            w_list = []
            b_list = []
            for i in range(len(x_test)):
                prediction, w, b = classification(x_train=x_train, y_train=y_train, x=x_test.iloc[i].values, alpha=omega)
                w_list.append(w)
                b_list.append(b)
                y_true = np.asarray(y_test)[i]
                if prediction >= 0.5 and y_true >= 0.5:
                    tp += 1
                elif prediction < 0.5 and y_true < 0.5:
                    tn += 1
                elif prediction < 0.5 and y_true >= 0.5:
                    fn += 1
                else:
                    fp += 1
                if prediction < 0.5:
                    prediction = 0
                else:
                    prediction = 1
                if act == 1:
                    predictions_for_answer.append(prediction)

        elif ml_model == "decision_tree":
            tree = decision_tree(x_train=x_train, y_train=y_train)
            predictions = []

            for i in range(len(x_test)):
                prediction = decision_tree(tree=tree, x=x_test.iloc[i].values)
                predictions.append(prediction)

            predictions = np.array(predictions) * params["y_std"] + params["y_mean"]
            if act == 1:
                new_prediction = decision_tree(tree=tree, x=x_new.iloc[0].values)
                new_prediction_real = (new_prediction * params["y_std"] + params["y_mean"])
                predictions_for_answer.append(new_prediction_real)
            y_test_real = np.array(y_test) * params["y_std"] + params["y_mean"]

            mse = np.mean((predictions - y_test_real) ** 2)
            rmse = np.sqrt(mse)
            ss_res = np.sum((y_test_real - predictions) ** 2)
            ss_tot = np.sum((y_test_real - np.mean(y_test_real)) ** 2)
            r2 = 1 - ss_res / ss_tot

            mse_avg.append(mse)
            rmse_avg.append(rmse)
            r2_avg.append(r2)

        else:
            random_number = np.random.randint(2, 30)
            trees = random_forest(x_train=x_train, y_train=y_train, n_trees=random_number) # Number of trees
            predictions = []

            for i in range(len(x_test)):
                prediction = random_forest(trees=trees, x=x_test.iloc[i].values)
                predictions.append(prediction)
            predictions = np.array(predictions)
            predictions_real = (predictions * params["y_std"] + params["y_mean"])
            y_test_real = (np.array(y_test) * params["y_std"] + params["y_mean"])
            if act == 1:
                new_prediction = random_forest(trees=trees, x=x_new.iloc[0].values)
                new_prediction_real = (new_prediction * params["y_std"] + params["y_mean"])
                predictions_for_answer.append(new_prediction_real)
            mse = np.mean((predictions_real - y_test_real) ** 2)
            rmse = np.sqrt(mse)
            ss_res = np.sum((y_test_real - predictions_real) ** 2)
            ss_tot = np.sum((y_test_real - np.mean(y_test_real)) ** 2)
            r2 = 1 - ss_res / ss_tot
            mse_avg.append(mse)
            rmse_avg.append(rmse)
            r2_avg.append(r2)

    if ml_model == "knn":
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        if act == 0:
            return [accuracy, precision, recall, f1]
        else:
            return predictions_for_answer, f1

    elif ml_model == "linear_regression":
        mse_avg = np.array(mse_avg).mean()
        rmse_avg = np.array(rmse_avg).mean()
        r2_avg = np.array(r2_avg).mean()
        if act == 0:
            return [mse_avg, rmse_avg, r2_avg]
        else:
            return w, b, mse_avg, x_new, params

    elif ml_model == "classification":
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        if act == 0:
            return [accuracy, precision, recall, f1]
        else:
            z = x_new @ w + b
            probability = 1 / (1 + np.exp(-z))
            return np.asarray(probability).reshape(-1), f1

    elif ml_model == "decision_tree":
        mse_avg_ = np.array(mse_avg).mean()
        rmse_avg_ = np.array(rmse_avg).mean()
        r2_avg_ = np.array(r2_avg).mean()
        if act == 0:
            return [mse_avg_, rmse_avg_, r2_avg_]
        else:
            return predictions_for_answer, mse_avg_

    else:
        mse_avg_ = np.array(mse_avg).mean()
        rmse_avg_ = np.array(rmse_avg).mean()
        r2_avg_ = np.array(r2_avg).mean()
        if act == 0:
            return [mse_avg_, rmse_avg_, r2_avg_]
        else:
            return predictions_for_answer, mse_avg_

def lifecycle(file_name, column_name, model_type, act, columns_name, alpha, omega, iterations, test_list=None):


    if str(type(file_name)) == "<class 'str'>":
        file = pd.read_csv(file_name).iloc[:-1].reset_index(drop=True)
    else:
        file = file_name.copy()
    file = file.drop(columns=columns_name or [])
    file = file.dropna(subset=[column_name])
    feature_columns = file.columns.drop(column_name)
    file = file.loc[file[feature_columns].isna().sum(axis=1) <= 1].reset_index(drop=True)

    classes = None
    if model_type == "classification":
        classes = sorted(file[column_name].unique().tolist())

        class_to_code = {classes[0]: 0, classes[1]: 1}
        file[column_name] = file[column_name].map(class_to_code)


    classification_models = ["knn", "classification"]
    linear_models = ["linear_regression", "decision_tree", "random_forest"]

    if model_type == "classification":
        metrics = []
        model_results = []
        for model_name in classification_models:
            if act == 0:
                accuracy, precision, recall, f1 = pipeline(column_=column_name, data=file, ml_model=model_name, act=0, alpha=alpha, omega=omega, iterations=iterations)
                metrics.append(f1)
                model_results.append([f"{model_name}:", accuracy, precision, recall, f1])
            else:
                if model_name == "knn":
                    prediction, f1 = pipeline(column_=column_name, data=file, ml_model=model_name, act=act, alpha=alpha, omega=omega, iterations=iterations, test_list=test_list)
                    metrics.append(f1)
                    model_results.append([f"{model_name}:", prediction])
                else:
                    probability, f1 = pipeline(column_=column_name, data=file, ml_model=model_name, act=act, alpha=alpha, omega=omega, iterations=iterations, test_list=test_list)
                    model_results.append([f"{model_name}:", probability])
                    metrics.append(f1)
        winner_index = np.argmax(np.array(metrics))
        winner = model_results[winner_index]
        if act == 0:
            return [model_results, "Winner: ", winner]
        if act == 1:
            return winner

    else:
        metrics = []
        model_results = []
        for model_name in linear_models:
            if act == 0:
                mse_avg_, rmse_avg_, r2_avg_ = pipeline(column_=column_name, data=file, ml_model=model_name, act=0, alpha=alpha, omega=omega, iterations=iterations)
                metrics.append(mse_avg_)
                model_results.append([f"{model_name}:", mse_avg_, rmse_avg_, r2_avg_])

            else:
                if model_name == "linear_regression":
                    if test_list is not None:
                        w, b, mse_avg_, x_new, params = pipeline(column_=column_name, data=file, ml_model=model_name, act=act, alpha=alpha, omega=omega, iterations=iterations, test_list=test_list)
                        model_results.append([f"{model_name}:", w, b, x_new, params])
                        metrics.append(mse_avg_)
                    else:
                        w, b, mse_avg_, x_new, params = pipeline(column_=column_name, data=file, ml_model=model_name, act=act, alpha=alpha, omega=omega, iterations=iterations)
                        metrics.append(mse_avg_)
                        model_results.append([f"{model_name}:", w, b, x_new, params])

                else:
                    if test_list is not None:
                        prediction, mse_avg_ = pipeline(column_=column_name, data=file, ml_model=model_name, act=act, alpha=alpha, omega=omega, iterations=iterations, test_list=test_list)
                        metrics.append(mse_avg_)
                        model_results.append([f"{model_name}:", prediction])
                    else:
                        prediction, mse_avg_ = pipeline(column_=column_name, data=file, ml_model=model_name, act=act, alpha=alpha, omega=omega, iterations=iterations)
                        metrics.append(mse_avg_)
                        model_results.append([f"{model_name}:", prediction])

        winner_index = np.argmin(np.array(metrics))
        winner = model_results[winner_index]

        if act == 0:
            return [model_results, "Winner: ", winner]

        if act == 1:
            return winner
