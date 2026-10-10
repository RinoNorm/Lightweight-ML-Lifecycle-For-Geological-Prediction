# Project Overview

This educational project was created to obtain predictions for small tabular datasets through a partial implementation of lightweight ML lifecycle. The project is under development and requires improvements to preprocessing and the elimination of potential data leakage.

## Motivation

The machine learning algorithms in this project were implemented from scratch without using `sklearn`, as this is an educational project designed to demonstrate the inner workings of training KNN, linear regression, classifiers, decision trees, and random forests.

## How It Works

This project implements a lightweight, partial MLOps workflow as follows: `__main__.py` calls the lifecycle through the `start` function, where data is loaded, passed to cross-validation (`cv`), and split into five folds.

Within each fold, the data goes through two processing stages: imputation and preprocessing. Depending on the target type, the imputer uses liner regression or knn to fills missing values in the file. There is one restriction: if an object has more than two missing features, it is removed from the training and test sets. Preprocessing follows.

A pipeline is then created within each fold to train and test the algorithms. Finally, the best model is selected based on the metrics and used to make the final prediction.

## Project Structure

The project consists of the following sequence of files:

`__main__.py` → `cli.py` → `pipeline.py` → `imputation.py` → `preprocessing.py` → `models.py`

| File | Role |
| --- | --- |
| `__main__.py` | Starts the project workflow. |
| `cli.py` | Defines the workflow parameters. |
| `pipeline.py` | Calls `imputation` and `preprocessing` to process the tabular dataset, then calls `models` to start training. |
| `imputation.py` | Handles missing values in the tabular dataset. |
| `preprocessing.py` | Preprocesses the tabular dataset. |
| `models.py` | Runs model training. |

## Supported Data and Usage

A sample of multiple objects is required, with different numerical features. The features of an object must differ from one another. The features of one object must differ from those of another object. For example, a sample of 15 objects with 8 features, where two objects have the following features:

| % | Content, g/t | C1 × % | % | Content, g/t | C2 × % | %C1 + %C2 | Gold content in the sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 35 | 0.05 | 1.8 | 62 | 0.05 | 3.1 | 4.9 | 0.05 |
| 29 | 0.27 | 8.0 | 71 | 3.68 | 260.4 | 268.4 | 2.68 |

Data with a small number of missing values in the table, as well as numerical data containing typos, is also suitable.

Text features are supported and encoded using one-hot encoding. For example, when predicting the price of an apartment, each city found in the training data gets its own column: 1 indicates that the apartment is located in that city, and 0 indicates that it is not. The categories are learned from the training portion of each fold, and the same columns are used for validation and new predictions. If a new object contains a category that was not present in the training data, all columns for that feature are set to 0.

Data with too many missing values is not suitable. Tables must have a structure with column names starting in cell A1, followed by features. Tables with gaps between columns are not suitable. Tables with too few objects or object features are not suitable (see Section 2). Tables containing data that is very similar from one object to another are not suitable, for example:

| No. | Profile | Station | Coordinates | Coordinates | ∆T, nT |
| --- | --- | --- | --- | --- | --- |
| No. | Profile | Station | X | Y | ∆T, nT |
| 1 | 2 | 3 | 4 | 5 | 6 |
| 1 | 6 | 200 | xxxxxxxx.x | xxxxxxxx.x | 4.2 |
| 2 | 6 | 205 | xxxxxxxx.x | xxxxxxxx.x | -11.8 |
| 3 | 6 | 210 | xxxxxxxx.x | xxxxxxxx.x | 0.5 |

### Installation

To run the model, you need Python and any CSV editor. Local MySQL databases can also be used.

Then install the following libraries in CMD: NumPy and pandas.

```bash
python -m pip install numpy
python -m pip install pandas
```

### Running the Project

Download the entire repository into a single folder. In the `start` function in `__main__.py`, specify where the dataset is stored: `database` or `csv`. Then run the project from IDE, which you chosen.

### In `Property.csv` you have to fill some information:

In 2 row you have to enter file name, which you want to use for training and prediction;

In 4 row you need to enter column name of value you want to predict;

In 6 row enter type of data, which you want to predict: class or property;

8 row uses to delete columns, what you needn't in prediction;

In 10 row enter new object properties without  columns from 4 and 8 rows.

> Prediction takes at least 10 minutes.

### Testing

To test the implementation, run `test.py`. Testing may take an hour and will report either MSE, RMSE, and R², or accuracy, precision, recall, and F1. If you need to run the test, you have to change one row from `pipeline.py`: 

row 11:
```bash
data = data.drop_duplicates().iloc[:-1].reset_index(drop=True)
```

If you needn't to run the test, change this row:

row 11:
```bash
data = data.drop_duplicates().reset_index(drop=True)
```

## Roadmap

1. Refine preprocessing to make it more flexible.
2. Identify and resolve potential data leakage.
3. Implement full database support.
4. Improve configuration and package structure.

## Metric Results

### Test 1

| Model | MSE | RMSE | R² |
| --- | --- | --- | --- |
| **linear_regression** | **0.0089** | **0.0939** | **0.997** |
| decision_tree | 0.2247 | 0.4721 | 0.9284 |
| random_forest | 0.1224 | 0.3408 | 0.9585 |

**Winner:** `linear_regression`

### Test 2

| Model | MSE | RMSE | R² |
| --- | --- | --- | --- |
| linear_regression | 4.9754 | 2.2204 | 0.5879 |
| **decision_tree** | **0.2322** | **0.221** | **0.9822** |
| random_forest | 0.2473 | 0.2858 | 0.981 |

**Winner:** `decision_tree`

### Test 3

| Model | MSE | RMSE | R² |
| --- | --- | --- | --- |
| linear_regression | 1.7153 | 1.3 | 0.4 |
| decision_tree | 0.7759 | 0.8771 | 0.7254 |
| **random_forest** | **0.5187** | **0.7168** | **0.8168** |

**Winner:** `random_forest`

### Test 4

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| **knn** | **0.7974** | **0.8089** | **0.8275** | **0.8181** |
| classification | 0.5063 | 0.5343 | 0.8045 | 0.6422 |

**Winner:** `knn`

### Test 5

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| knn | 0.9303 | 0.948 | 0.9125 | 0.9299 |
| **classification** | **0.981** | **0.9753** | **0.9875** | **0.9813** |

**Winner:** `classification`

## Numerical and Class Prediction Results

The values below are predictions of either the amount of gold in the rock or the type of rock itself.

| Test | Actual value | Predicted value |
| --- | --- | --- |
| 1 | 5.7453 | 5.8117 |
| 2 | 1.0034 | 0.9996 |
| 3 | 3.4465 | 3.238 |
| 4 | background | 0 |
| 5 | sulfide | 1 |
