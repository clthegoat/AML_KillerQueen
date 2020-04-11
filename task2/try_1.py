import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import tqdm
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

import xgboost as xgb
model_XGBoostClassifier = xgb.XGBClassifier()
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight


def bas(y_pred, y_true):
    return balanced_accuracy_score(y_true, y_pred)


def load_data(x_path='./X_train.csv', y_path='./y_train.csv', x_test_path='./X_test.csv'):
    """
    Load data from .csv files
    :param x_path: relative path of x
    :param y_path: relative path of y
    :param x_test_path :relative path of x_test
    :return data_x, data_y, data_x_test: X, Y, X_test in pd.DataFrame format
    """
    print()
    print("Loading data from {}, {} and {}".format(x_path, y_path, x_test_path))
    data_x = pd.read_csv(x_path)
    data_y = pd.read_csv(y_path)
    data_x_test = pd.read_csv(x_test_path)
    print('Data loaded, data_set Information:')
    print("x: {}".format(data_x.shape))
    print("y: {}".format(data_y.shape))
    print("x_test: {}".format(data_x_test.shape))
    print()
    return data_x, data_y, data_x_test


def from_csv_to_ndarray(data):
    """
    Fransfer data from pd.DataFrame to ndarray for later model training
    :param data: data in pd.DataFrame
    :return ndarray: data in ndarray
    """
    data.head()
    ndarray = data.values
    if ndarray.shape[1] == 2:
        return ndarray[:, 1]
    else:
        return ndarray[:, 1:]


def data_preprocessing(x_raw, x_test_raw):
    """
    Data preprocessing including normalization or scaling, etc.
    :param x_raw: np.ndarray, data before preprocessing
    :param x_test_raw: np.ndarray, test data before preprocessing
    :return x_after, x_test_after: data after preprocessing
    """
    std = StandardScaler()
    x_concat = np.concatenate((x_raw, x_test_raw), axis=0)
    x_after = std.fit_transform(x_concat)
    return x_after[:len(x_raw)], x_after[len(x_raw):]


def over_sampling(x_train, y_train):
    print()
    print("Doing over sampling...")
    print("Before over sampling:")
    class0_num = np.sum(y_train == 0)
    class1_num = np.sum(y_train == 1)
    class2_num = np.sum(y_train == 2)
    print("#Sample in Class 0: {}".format(class0_num))
    print("#Sample in Class 1: {}".format(class1_num))
    print("#Sample in Class 2: {}".format(class2_num))
    x_out = x_train
    y_out = y_train

    print("After over sampling:")
    class0_num = np.sum(y_out == 0)
    class1_num = np.sum(y_out == 1)
    class2_num = np.sum(y_out == 2)
    print("#Sample in Class 0: {}".format(class0_num))
    print("#Sample in Class 1: {}".format(class1_num))
    print("#Sample in Class 2: {}".format(class2_num))

    return x_out, y_out


def select_feature(x_train, y_train, x_test):
    """
    Select features based on training data(but actually we can use all data)
    :param x_train: features
    :param y_train: labels
    :param x_test: test features
    :return x_selected, y_train, x_test_selected: return selected feature and label
    """
    print()
    print("Selecting features...")
    print("Before feature selection: {}".format(x_train.shape))
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, class_weight="balanced")
    random_feature = np.random.rand(x_train.shape[0]*5).reshape((x_train.shape[0], 5))
    x_train_new = np.concatenate((x_train, random_feature), axis=1)
    clf.fit(x_train_new, y_train)
    random_feature_importance = clf.feature_importances_[-5:]
    feature_importance = clf.feature_importances_[:-5]
    for i in range(5):
        feature_importance[feature_importance < random_feature_importance[i]] = 0
    select_idx = feature_importance > 0
    x_selected = x_train[:, select_idx]
    x_test_selected = x_test[:, select_idx]
    print("After feature selection: {}".format(x_selected.shape))
    return x_selected, y_train, x_test_selected


def try_different_method(model, x_train, y_train, x_test, y_test):
    """
    Inner function in train_evaluate_return_best_model for model training.
    :param model: one specific model
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return score:
    """
    model.fit(x_train, y_train)
    result_test = model.predict(x_test)
    result_train = model.predict(x_train)
    score_test = bas(result_test, y_test)
    score_train = bas(result_train, y_train)
    return score_test, score_train


def evaluate_model(model, x_all, y_all):
    print("Evaluating model...")
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True)
    score_mean_test = 0
    score_mean_train = 0
    for train_idx, test_idx in kf.split(x_all):
        instance_model = clone(model)
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        x_test = x_all[test_idx]
        y_test = y_all[test_idx]
        score_test, score_train = try_different_method(instance_model, x_train, y_train, x_test, y_test)
        score_mean_test += score_test
        score_mean_train += score_train

    score_mean_test /= n_folds
    score_mean_train /= n_folds
    print("Mean score on test set: {}".format(score_mean_test))
    print("Mean score on train set: {}".format(score_mean_train))
    model.fit(x_all, y_all)
    return model


def main():
    print()
    print('***************By Killer Queen***************')

    # configs:
    X_train_dir = './data/X_train.csv'
    y_train_dir = './data/y_train.csv'
    X_test_dir = './data/X_test.csv'
    y_pred_save_dir = './y_test_try.csv'
    do_over_sampling = False
    data_x, data_y, data_x_test = load_data(x_path=X_train_dir, y_path=y_train_dir, x_test_path=X_test_dir)
    test_ID = data_x_test['id']
    y_train = from_csv_to_ndarray(data=data_y)
    x_train = from_csv_to_ndarray(data=data_x)
    x_test = from_csv_to_ndarray(data=data_x_test)
    x_train, x_test = data_preprocessing(x_train, x_test)   # Normalizaiton and ...
    if do_over_sampling:
        x_train, y_train = over_sampling(x_train, y_train)
    x_train, y_train, x_test = select_feature(x_train, y_train, x_test)

    train = np.concatenate((x_train, y_train.reshape((-1, 1))), axis=1)
    df = pd.DataFrame(train)
    corrmat = df.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()

    from sklearn.svm import SVC
    clf = SVC(gamma='auto', class_weight='balanced')
    model = evaluate_model(clf, x_train, y_train)
    prediction = model.predict(x_test)
    # show correlation heat map

    sub = pd.DataFrame()
    sub['id'] = test_ID
    sub['y'] = prediction
    sub.to_csv(y_pred_save_dir, index=False)


main()
