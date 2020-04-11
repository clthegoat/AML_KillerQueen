import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

from pyod.utils.utility import standardizer
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.xgbod import XGBOD


###########Add different models here!!!!###########
model_heads = []
models = []
from sklearn import tree  # 0

model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
model_heads.append("Decision Tree Regression\t\t")
models.append(model_DecisionTreeRegressor)

from sklearn import linear_model  # 1

model_LinearRegression = linear_model.LinearRegression()
model_heads.append("Linear Regression\t\t\t\t")
models.append(model_LinearRegression)

from sklearn import svm  # 2

model_SVR = svm.SVR()
model_heads.append("Support Vector Machine Regression")
models.append(model_SVR)

from sklearn import neighbors  # 3

model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
model_heads.append("K-Nearest Neighbor Regression\t")
models.append(model_KNeighborsRegressor)

from sklearn import ensemble  # 4

model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)
model_heads.append("Random Forest Regression\t\t")
models.append(model_RandomForestRegressor)

from sklearn import ensemble  # 5

model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=150)
model_heads.append("AdaBoost Regression\t\t\t\t")
models.append(model_AdaBoostRegressor)

from sklearn import ensemble  # 6

model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=200)
model_heads.append("Gradient Boosting Regression\t")
models.append(model_GradientBoostingRegressor)

from sklearn.ensemble import BaggingRegressor  # 7

model_BaggingRegressor = BaggingRegressor()
model_heads.append("Bagging Regression\t\t\t\t")
models.append(model_BaggingRegressor)

from sklearn.tree import ExtraTreeRegressor  # 8

model_ExtraTreeRegressor = ExtraTreeRegressor()
model_heads.append("ExtraTree Regression\t\t\t")
models.append(model_ExtraTreeRegressor)

import xgboost as xgb  # 9

model_XGBoostRegressor = xgb.XGBRegressor(n_estimators=200)
model_heads.append("XGBoost Regression\t\t\t\t")
models.append(model_XGBoostRegressor)


##########Model Adding Ends###########


def load_data(x_path='./X_train.csv', y_path='./y_train.csv'):
    """
    Load data from .csv files
    :param x_path: relative path of x
    :param y_path: relative path of y
    :return data_x, data_y: X, Y in pd.DataFrame format
    """
    print()
    print("Loading data from {} and {}".format(x_path, y_path))
    data_x = pd.read_csv(x_path)
    data_y = pd.read_csv(y_path)
    print('Data loaded, data_set Information:')
    print("x: {}".format(data_x.shape))
    print("y: {}".format(data_y.shape))
    print()
    return data_x, data_y


def fill_missing_data(data_x):
    """
    Fill Nan value in data of pd.DataFrame format
    :param data_x: feature or data in pd.DataFrame format
    :return data_x_filled: filled data
    """
    print("Filling missing data...")
    data_x_filled = data_x.fillna(data_x.median())
    return data_x_filled


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


def data_preprocessing(x_raw):
    """
    Data preprocessing including normalization or scaling, etc.
    :param x_raw: np.ndarray, data before preprocessing
    :return x_after: data after preprocessing
    """
    std = StandardScaler()
    x_after = std.fit_transform(x_raw)
    # x_after = standardizer(x_raw)
    return x_after


def feature_selection(x_train, y_train):
    """
    Select features based on training data(but actually we can use all data)
    :param x_train: features
    :param y_train: labels
    :return x_selected, y_train, model_select: return selected feature and label with feature selection model
    (sklearn.feature_selection.SelectFromModel)
    """
    print()
    print("Selecting features...")
    print("Before feature selection: {}".format(x_train.shape))
    clf_selec = linear_model.Lasso(alpha=0.1)
    model_select = SelectFromModel(clf_selec.fit(x_train, y_train), prefit=True)
    x_selected = model_select.transform(x_train)
    print("After feature selection: {}".format(x_selected.shape))
    return x_selected, y_train, model_select


def del_rowsorcolumns(X_in, idx, axis):
    X_out = np.delete(X_in, idx, axis=axis)
    return X_out


def outlier_detection(x_raw, y_raw):
    """
    Filter all ourlier points
    :param x_raw: feature in ndarray
    :param y_raw: label in ndarray
    :return x_clean, y_clean: cleaned feature and label in ndarray
    """
    # TODO Filter the outliers.
    print()
    print("Detecting outliers...")
    print("Before outlier detection: {}".format(x_raw.shape))
    outliers_fraction = 0.04
    random_state = np.random.RandomState(42)
    # all outlier detection method candidate list as follows
    classifiers = {'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
                   'Cluster-based Local Outlier Factor': CBLOF(contamination=outliers_fraction, check_estimator=False,
                                                               random_state=random_state),
                   'Feature Bagging': FeatureBagging(contamination=outliers_fraction, random_state=random_state),
                   'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
                   'Isolation Forest': IForest(contamination=outliers_fraction, random_state=random_state),
                   'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
                   'Local Outlier Factor (LOF)': LOF(contamination=outliers_fraction),
                   'Minimum Covariance Determinant (MCD)': MCD(contamination=outliers_fraction,
                                                               random_state=random_state),
                   'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
                   'Principal Component Analysis (PCA)': PCA(contamination=outliers_fraction,
                                                             random_state=random_state),
                   'Improving Supervised Outlier Detection with Unsupervised Representation Learning': XGBOD(contamination=outliers_fraction),
                   }
    clf_name = 'Isolation Forest'
    clf = IForest(contamination=outliers_fraction, random_state=random_state)
    # clf_name = 'Angle-based Outlier Detector (ABOD)'
    # clf = ABOD(contamination=outliers_fraction, method='default')
    clf.fit(x_raw)
    y_pred = clf.predict(x_raw)
    # for pyod, 1 means outliers and 0 means inliers
    # for sklearn,  -1 means outliers and 1 means inliers
    idx_y_pred = [i for i in range(0, 1212) if y_pred[i] == 1]
    x_clean = del_rowsorcolumns(x_raw, idx_y_pred, axis=0)
    y_clean = del_rowsorcolumns(y_raw, idx_y_pred, axis=0)
    print("After outlier detection: {}".format(x_clean.shape))
    assert (x_clean.shape[0] == y_clean.shape[0])
    return x_clean, y_clean
    # return y_pred, idx_y_pred


def try_different_method(model, x_train, y_train, x_test, y_test, score_func):
    """
    Inner function in train_evaluate_return_best_model for model training.
    :param model: one specific model
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param score_func:
    :return score:
    """
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    return score_func(y_test, result)


def train_evaluate_return_best_model(x_all, y_all, score_func=r2_score):
    """
    Train predefined models on data using 5-fold validation
    :param x_all: ndarray containing all features
    :param y_all: ndarray containing all labels
    :param score_func: score function
    :return best_model: best model trained on all data
    """
    print()
    print("Training model with K-fords...")
    kf = KFold(n_splits=5, shuffle=True)
    best_score = 0
    best_idx = 0
    for (model_idx, model) in enumerate(models):
        score_mean = 0
        for train_idx, test_idx in kf.split(x_all):
            x_train = x_all[train_idx]
            y_train = y_all[train_idx]
            x_test = x_all[test_idx]
            y_test = y_all[test_idx]
            score_mean += try_different_method(model, x_train, y_train, x_test, y_test, score_func)
        score_mean /= 5
        print("{} \t score: {}".format(model_heads[model_idx], score_mean))
        if best_score < score_mean:
            best_score = score_mean
            best_idx = model_idx
    print("Training done")
    print("Best model: {}\t Score: {}".format(model_heads[best_idx], best_score))
    best_model = models[best_idx]
    best_model.fit(x_all, y_all)
    return best_idx, best_model


def get_model(x_all, y_all, model_idx):
    """
    Given model index return the corresponding model trained on all data
    :param x_all:
    :param y_all:
    :param model_idx:
    :return model:
    """
    print()
    print("Training with all data using {}".format(model_heads[model_idx]))
    model = models[model_idx].fit(x_all, y_all)
    return model


def predict_and_save_results(model, test_path, save_path, model_select):
    print()
    print("Load test data from {}".format(test_path))
    x_new = pd.read_csv(test_path)
    x_new = fill_missing_data(x_new)
    x_new.head()
    ndarray = x_new.values
    ids = ndarray[:, 0]
    x_new = ndarray[:, 1:]
    x_new = data_preprocessing(x_new)
    x_new = model_select.transform(x_new)
    y_pred = model.predict(x_new)
    out = np.zeros((len(y_pred), 2))
    out[:, 0] = ids
    out[:, 1] = y_pred
    print("Prediction saved to {}".format(save_path))
    pd.DataFrame(out, columns=['id', 'y']).to_csv(save_path)


def main():
    print()
    print('***************By Killer Queen***************')
    data_x, data_y = load_data(x_path='./X_train.csv', y_path='./y_train.csv')
    data_x_filled = fill_missing_data(data_x=data_x)
    x_ndarray = from_csv_to_ndarray(data=data_x_filled)
    y_ndarray = from_csv_to_ndarray(data=data_y)
    x_ndarray = data_preprocessing(x_raw=x_ndarray)
    x_clean, y_clean = outlier_detection(x_raw=x_ndarray, y_raw=y_ndarray)
    x_select, y_select, model_select = feature_selection(x_train=x_clean, y_train=y_clean)
    # x_clean, y_clean = outlier_detection(x_raw=x_select, y_raw=y_select)
    score_function = r2_score
    find_best_model = True  # Change this to false and set a model_idx to train a specific model!!!
    model_idx = 9
    if find_best_model:
        model_idx, best_model = train_evaluate_return_best_model(x_all=x_select, y_all=y_select,
                                                                 score_func=score_function)
    else:
        best_model = get_model(x_all=x_select, y_all=y_select, model_idx=model_idx)
    print()
    print("Using model: {}".format(model_heads[model_idx]))
    predict_and_save_results(model=best_model, test_path='./X_test.csv',
                             save_path='./y_test.csv', model_select=model_select)


if __name__ == '__main__':
    main()
