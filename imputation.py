import pandas as pd
from models import knn, linear_regression
from preprocessing import preprocessing

def imputer(file_name, columns_name, alpha, iterations, w_and_b=None):
    if str(type(file_name)) == "<class 'str'>":
        data = pd.read_csv(file_name)
    else:
        data = file_name.copy()

    data = data.drop(columns=columns_name or [])

    if w_and_b is None:
        limit_time_data = data.dropna().reset_index(drop=True)
        w_and_b = {}

        for c in data.columns:
            x_train, y_train, params = preprocessing(data=limit_time_data, c=c)
            placeholder = limit_time_data[c].iloc[0]

            if "y_mean" in params:
                w, b = linear_regression(x_train=x_train, y_train=y_train, alpha=alpha, iterations=iterations)
                w_and_b[c] = [w, b, params, placeholder]
            else:
                w_and_b[c] = [x_train, y_train, params, placeholder]

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if pd.isna(data.iloc[i, j]):
                c = data.columns[j]
                first, second, params, placeholder = w_and_b[c]

                row = data.iloc[[i]].copy()
                row[c] = placeholder
                x, _, _ = preprocessing(data=row, c=c, params=params)

                if "y_mean" in params:
                    w, b = first, second
                    value = (x.to_numpy() @ w + b).item()
                    value = value * params["y_std"] + params["y_mean"]
                else:
                    x_train, y_train = first, second
                    k = min(5, len(x_train))
                    class_code = knn(x_train=x_train, y_train=y_train, x=x.iloc[0].to_numpy(), k=k)
                    value = params["classes"][int(class_code)]

                data.iloc[i, j] = value

    return data, w_and_b
