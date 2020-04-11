import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
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
from sklearn import tree    # 0
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
model_heads.append("Decision Tree Regression\t\t")
models.append(model_DecisionTreeRegressor)

from sklearn import linear_model    # 1
model_LinearRegression = linear_model.LinearRegression()
model_heads.append("Linear Regression\t\t\t\t")
models.append(model_LinearRegression)

from sklearn import svm     # 2
model_SVR = svm.SVR()
model_heads.append("Support Vector Machine Regression")
models.append(model_SVR)

from sklearn import neighbors   # 3
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
model_heads.append("K-Nearest Neighbor Regression\t")
models.append(model_KNeighborsRegressor)

from sklearn import ensemble    # 4
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)
model_heads.append("Random Forest Regression\t\t")
models.append(model_RandomForestRegressor)

from sklearn import ensemble    # 5
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=150)
model_heads.append("AdaBoost Regression\t\t\t\t")
models.append(model_AdaBoostRegressor)

from sklearn import ensemble    # 6
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor()
model_heads.append("Gradient Boosting Regression\t")
models.append(model_GradientBoostingRegressor)

from sklearn.ensemble import BaggingRegressor   # 7
model_BaggingRegressor = BaggingRegressor()
model_heads.append("Bagging Regression\t\t\t\t")
models.append(model_BaggingRegressor)

from sklearn.tree import ExtraTreeRegressor     # 8
model_ExtraTreeRegressor = ExtraTreeRegressor()
model_heads.append("ExtraTree Regression\t\t\t")
models.append(model_ExtraTreeRegressor)

import xgboost as xgb       # 9
model_XGBoostRegressor = xgb.XGBRegressor()
model_heads.append("XGBoost Regression\t\t\t\t")
models.append(model_XGBoostRegressor)
##########Model Adding Ends###########


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


def filter_low_var_feature(x, threshold=1e-10):
    x += 1e-7
    x_var = np.var(x / np.mean(x, axis=0), axis=0)
    x_idx = x_var > threshold
    x -= 1e-7
    return x[:, x_idx], x_idx


def fill_missing_data(data_x, data_x_test, load_file=True, filling_method="random_forest"):
    """
    Fill Nan value in data of pd.DataFrame format.
    !!! Normalization moved to this part
    :param data_x: feature or data in pd.DataFrame format
    :param data_x_test: test feature or data in pd.DataFrame format
    :param load_file: Load from pre-filled data?
    :param filling_method: filling method
    :return data_x_filled, data_x_test_filled: !!!normalized filled data in ndarray
    """
    print("Filling missing data with {}".format(filling_method))
    if filling_method == 'pandas_median':
        data_x_filled = data_x.fillna(data_x.median())
        data_x_test_filled = data_x_test.fillna(data_x_test.median())
        std = StandardScaler()
        x_concat = np.concatenate((from_csv_to_ndarray(data=data_x_filled),
                                   from_csv_to_ndarray(data=data_x_test_filled)), axis=0)
        x_filtered, _ = filter_low_var_feature(x_concat, 1e-10)
        x_after = std.fit_transform(x_filtered)
        return x_after[:len(data_x_filled)], x_after[len(data_x_filled):]
    elif filling_method == 'similarity_matrix':
        if load_file:
            return np.load('./x_filled.npy'), np.load('./x_test_filled.npy')
            # Step 1: fill data using mean
        data_x_filled = data_x.fillna(data_x.median())
        x_ori = from_csv_to_ndarray(data=data_x)
        x_filled = from_csv_to_ndarray(data=data_x_filled)
        data_x_test_filled = data_x_test.fillna(data_x_test.median())
        x_test_ori = from_csv_to_ndarray(data=data_x_test)
        x_test_filled = from_csv_to_ndarray(data=data_x_test_filled)
        x_concat = np.concatenate((x_ori, x_test_ori), axis=0)
        x_filled_concat = np.concatenate((x_filled, x_test_filled), axis=0)
        missing_idx = np.isnan(x_concat)
        # normalization
        std = StandardScaler()

        x_filled_concat, idx = filter_low_var_feature(x_filled_concat, 1e-10)
        x_filled_concat = std.fit_transform(x_filled_concat)

        missing_idx = missing_idx[:, idx]

        # Step 2: fill using similarity matrix
        for i in range(3):
            similarity_matrix = cosine_similarity(x_filled_concat, x_filled_concat)
            similarity_matrix[similarity_matrix > 0.999] = 0
            similarity_matrix[similarity_matrix < 0] = 0
            for j in tqdm.tqdm(range(len(x_filled_concat))):
                weighted_sum = np.sum(similarity_matrix[:, j].reshape([-1, 1]) * x_filled_concat, axis=0) / \
                               np.sum(similarity_matrix[:, j])
                x_filled_concat[j][missing_idx[j]] = weighted_sum[missing_idx[j]]
        np.save('x_filled.npy', x_filled_concat[:len(x_filled)])
        np.save('x_test_filled.npy', x_filled_concat[len(x_filled):])
        return x_filled_concat[:len(x_filled)], x_filled_concat[len(x_filled):]
    elif filling_method == "random_forest":
        if load_file:
            return np.load('./x_filled_rf2.npy'), np.load('./x_test_filled_rf2.npy')
        data_x_filled = data_x.fillna(data_x.median())
        x_ori = from_csv_to_ndarray(data=data_x)
        x_filled = from_csv_to_ndarray(data=data_x_filled)
        data_x_test_filled = data_x_test.fillna(data_x_test.median())
        x_test_ori = from_csv_to_ndarray(data=data_x_test)
        x_test_filled = from_csv_to_ndarray(data=data_x_test_filled)
        x_concat = np.concatenate((x_ori, x_test_ori), axis=0)
        x_filled_concat = np.concatenate((x_filled, x_test_filled), axis=0)
        missing_idx = np.isnan(x_concat)
        # normalization
        std = StandardScaler()
        x_filled_concat, idx = filter_low_var_feature(x_filled_concat, 1e-10)
        x_filled_concat = std.fit_transform(x_filled_concat)

        missing_idx = missing_idx[:, idx]
        feature_len = x_filled_concat.shape[1]
        for j in range(2):
            for i in tqdm.tqdm(range(feature_len)):
                train = x_filled_concat[missing_idx[:, i] == 0]
                test = x_filled_concat[missing_idx[:, i] == 1]
                x_train = train[:, np.concatenate((np.arange(i), np.arange(i + 1, feature_len)))]
                y_train = train[:, i]
                x_test = test[:, np.concatenate((np.arange(i), np.arange(i + 1, feature_len)))]
                rfr = ensemble.RandomForestRegressor(n_estimators=20)
                rfr.fit(x_train, y_train)
                x_filled_concat[missing_idx[:, i] == 1, i] = rfr.predict(x_test)
        np.save('x_filled_rf2.npy', x_filled_concat[:len(x_filled)])
        np.save('x_test_filled_rf2.npy', x_filled_concat[len(x_filled):])
        return x_filled_concat[:len(x_filled)], x_filled_concat[len(x_filled):]


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


def feature_selection(x_train, y_train, x_test):
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
    clf_selec = linear_model.Lasso(alpha=0.1)
    model_select = SelectFromModel(clf_selec.fit(x_train, y_train), prefit=True)
    x_selected = model_select.transform(x_train)
    x_test_select = model_select.transform(x_test)
    print("After feature selection: {}".format(x_selected.shape))
    return x_selected, y_train, x_test_select


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
    result_test = model.predict(x_test)
    result_train = model.predict(x_train)
    return score_func(y_test, result_test), score_func(y_train, result_train)


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
        score_mean_test = 0
        score_mean_train = 0
        for train_idx, test_idx in kf.split(x_all):
            x_train = x_all[train_idx]
            y_train = y_all[train_idx]
            x_test = x_all[test_idx]
            y_test = y_all[test_idx]
            score_test, score_train = try_different_method(model, x_train, y_train, x_test, y_test, score_func)
            score_mean_test+=score_test
            score_mean_train+=score_train
        score_mean_test /= 5
        score_mean_train /= 5
        print("{} \t score train: {}, score test: {}".format(model_heads[model_idx], score_mean_train, score_mean_test))
        if best_score < score_mean_test:
            best_score = score_mean_test
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


def predict_and_save_results(model, x_test, save_path):
    print()
    y_pred = model.predict(x_test)
    out = np.zeros((len(y_pred), 2))
    ids = np.arange(0, len(x_test))
    out[:, 0] = ids
    out[:, 1] = y_pred
    print("Prediction saved to {}".format(save_path))
    pd.DataFrame(out, columns=['id', 'y']).to_csv(save_path)


def main():
    print()
    print('***************By Killer Queen***************')
    data_x, data_y, data_x_test = load_data(x_path='./X_train.csv', y_path='./y_train.csv', x_test_path='./X_test.csv')
    x_ndarray, x_test_ndarray = fill_missing_data(data_x=data_x, data_x_test=data_x_test,
                                                  load_file=True, filling_method="pandas_median")
    y_ndarray = from_csv_to_ndarray(data=data_y)
    x_clean, y_clean = outlier_detection(x_raw=x_ndarray, y_raw=y_ndarray)
    x_select, y_select, x_test_select = feature_selection(x_train=x_clean, y_train=y_clean, x_test=x_test_ndarray)
    score_function = r2_score
    find_best_model = True     # Change this to false and set a model_idx to train a specific model!!!
    model_idx = 8
    if find_best_model:
        model_idx, best_model = train_evaluate_return_best_model(x_all=x_select, y_all=y_select, score_func=score_function)
    else:
        best_model = get_model(x_all=x_select, y_all=y_select, model_idx=model_idx)
    print()
    print("Using model: {}".format(model_heads[model_idx]))
    predict_and_save_results(model=best_model, x_test=x_test_select, save_path='./y_test.csv')


main()
