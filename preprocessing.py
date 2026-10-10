import numpy as np
import pandas as pd

def preprocessing(data, c, params=None, columns_name=None, scale_y=True):
    data = pd.DataFrame(data)
    if columns_name is not None:
        data = data.drop(columns=columns_name)
    data = data.drop_duplicates().reset_index(drop=True)

    # Cleaning
    for i in range(data.shape[1]):
        column = data.iloc[:, i]
        numeric = pd.to_numeric(column, errors="coerce")

        invalid = column.notna() & numeric.isna()
        if numeric.notna().any() and not invalid.any():
            data[data.columns[i]] = numeric

        if numeric.notna().any() and invalid.any():
            error_column_name = data.columns[i]

            for index in column[invalid].index:
                error_ = np.array(list(str(column.loc[index])))
                error_ = np.array([x for x in error_ if x.isdigit() or x == "." or x == "-"])
                number = float("".join(error_))
                numeric.loc[index] = number

            data[error_column_name] = numeric

    x = data.drop(columns=[c])
    y = data[c]

    # Numerical and categorical features
    if params is None:
        x_numerical = x.select_dtypes(include="number").copy()
        x_categorical = x.select_dtypes(include=["object", "string", "category"]).astype("string").copy()
    else:
        numerical_columns = params["mean"].index.tolist()
        categorical_columns = list(params["categories"])

        x_numerical = x[numerical_columns].apply(pd.to_numeric, errors="raise").copy()
        x_categorical = x[categorical_columns].astype("string").copy()

    # Types and categories
    y_is_numeric = pd.api.types.is_numeric_dtype(y)

    if not x_numerical.empty:
        x_numerical = x_numerical.astype(float)
        x_numerical = x_numerical.round(2)

    if not x_categorical.empty:
        x_categorical = x_categorical.apply(lambda col: col.str.lower())

    # Scaling
    if params is None:
        params = {}

        params["mean"] = x_numerical.mean()
        params["std"] = x_numerical.std(ddof=0)
        params["std"] = params["std"].replace(0, 1)

        if y_is_numeric and scale_y:
            params["y_mean"] = y.mean()
            params["y_std"] = y.std(ddof=0)

            if params["y_std"] == 0:
                params["y_std"] = 1

    if not x_numerical.empty:
        x_numerical = (x_numerical - params["mean"]) / params["std"]

    if y_is_numeric and scale_y:
        y = (y - params["y_mean"]) / params["y_std"]

    # Encoding
    if "categories" not in params:
        params["categories"] = {col: x_categorical[col].dropna().unique().tolist() for col in x_categorical.columns}

    for col, categories in params["categories"].items():
        x_categorical[col] = pd.Categorical(x_categorical[col], categories=categories)

    if not x_categorical.empty:
        x_categorical = pd.get_dummies(x_categorical, dtype=int)

    x = pd.concat([x_numerical, x_categorical], axis=1)

    if "feature_columns" not in params:
        params["feature_columns"] = x.columns.tolist()

    x = x.reindex(columns=params["feature_columns"], fill_value=0)

    if not pd.api.types.is_numeric_dtype(y):
        if "classes" not in params:
            y, params["classes"] = pd.factorize(y)
        else:
            y = pd.Categorical(y, categories=params["classes"]).codes

    return x, y, params
