import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
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
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


# =============Add different models here!!!!=============
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
# params = {'learning_rate': 0.1, 'n_estimators': 400, 'max_depth': 8, 'min_child_weight': 2, 'seed': 0,
#           'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 3, 'reg_lambda': 2}
model_XGBoostRegressor = xgb.XGBRegressor()
model_heads.append("XGBoost Regression\t\t\t\t")
models.append(model_XGBoostRegressor)
# =============Model Adding Ends=============

# =============For Esemble and Stacking =============
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from sklearn import linear_model

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                            max_depth=4, max_features='sqrt',
                                            min_samples_leaf=15, min_samples_split=10,
                                            loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


def get_model_score(model, x_all, y_all, n_folds=5):
    score_func = r2_score
    kf = KFold(n_splits=n_folds, shuffle=True)
    score_mean_test = 0
    score_mean_train = 0
    for train_idx, test_idx in kf.split(x_all):
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        x_test = x_all[test_idx]
        y_test = y_all[test_idx]
        score_test, score_train = try_different_method(model, x_train, y_train, x_test, y_test, score_func)
        score_mean_test += score_test
        score_mean_train += score_train
    score_mean_test /= n_folds
    score_mean_train /= n_folds
    return score_mean_test


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    """
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    """
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


# =============For Esemble and Stacking(end)=============

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


def fill_missing_data(data_x, data_x_test, load_file=True, filling_method="random_forest",
                      train_file_dir='./train_file_dir_default.npy', test_file_dir='./test_file_dir_default.npy',
                      pre_select_idx_dir="./pre_select_idx_dir_default.npy",
                      train_save_dir='./train_save_dir_default.npy',
                      test_save_dir='./test_save_dir_default.npy'):
    """
    Fill Nan value in data of pd.DataFrame format.
    !!! Normalization moved to this part
    :param data_x: feature or data in pd.DataFrame format
    :param data_x_test: test feature or data in pd.DataFrame format
    :param load_file: Load from pre-filled data?
    :param filling_method: filling method select from "pandas_median"/"random_forest"/"similarity_matrix"
    :param train_file_dir: pre_filled train feature dir
    :param test_file_dir: pre_filled test feature dir
    :param train_save_dir: if load_file is False, generated filled train feature dir
    :param test_save_dir: if load_file is False, generated filled test feature dir
    :param pre_select_idx_dir: pre-selected feature index file dir
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
            return np.load(train_file_dir), np.load(test_file_dir)
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
        np.save(train_save_dir, x_filled_concat[:len(x_filled)])
        np.save(test_save_dir, x_filled_concat[len(x_filled):])
        return x_filled_concat[:len(x_filled)], x_filled_concat[len(x_filled):]
    elif filling_method == "random_forest":
        if load_file:
            return np.load(train_file_dir), np.load(test_file_dir)
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
        pre_selected_idx = np.load(pre_select_idx_dir)
        x_filled_concat = x_filled_concat[:, pre_selected_idx]
        missing_idx = missing_idx[:, pre_selected_idx]
        feature_len = x_filled_concat.shape[1]
        for j in range(3):
            for i in tqdm.tqdm(range(feature_len)):
                train = x_filled_concat[missing_idx[:, i] == 0]
                test = x_filled_concat[missing_idx[:, i] == 1]
                x_train = train[:, np.concatenate((np.arange(i), np.arange(i + 1, feature_len)))]
                y_train = train[:, i]
                x_test = test[:, np.concatenate((np.arange(i), np.arange(i + 1, feature_len)))]
                rfr = ensemble.RandomForestRegressor(n_estimators=20)
                rfr.fit(x_train, y_train)
                x_filled_concat[missing_idx[:, i] == 1, i] = rfr.predict(x_test)
        np.save(train_save_dir, x_filled_concat[:len(x_filled)])
        np.save(test_save_dir, x_filled_concat[len(x_filled):])
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


def generate_preselect_idx(x_train_dir, x_test_dir, y_dir, feature_num=100, save_dir='./default_preselect_idx.npy'):
    """
    Select features based on training data(but actually we can use all data)
    :param x_train_dir: train.csv file dir
    :param x_test_dir: test.csv file dir
    :param y_dir: label.csv file dir
    :param feature_num: selected feature number
    :param save_dir: selected index save dir
    """
    print()
    print("Generating feature pre_select_idx...")
    data_x = pd.read_csv(x_train_dir)
    data_x_test = pd.read_csv(x_test_dir)
    data_x_filled = data_x.fillna(data_x.median())
    data_x_test_filled = data_x_test.fillna(data_x_test.median())
    std = StandardScaler()
    x_concat = np.concatenate((from_csv_to_ndarray(data=data_x_filled),
                               from_csv_to_ndarray(data=data_x_test_filled)), axis=0)
    x_filtered, _ = filter_low_var_feature(x_concat, 1e-10)
    x_after = std.fit_transform(x_filtered)
    x_train = x_after[:len(data_x_filled)]
    y_train = from_csv_to_ndarray(pd.read_csv(y_dir))
    print("Before feature pre-selection: {}".format(x_train.shape))
    from sklearn.feature_selection import f_classif
    from sklearn.feature_selection import SelectKBest
    model_select = SelectKBest(score_func=f_classif, k=feature_num)
    model_select.fit(x_train, y_train)
    selected_index = np.arange(x_train.shape[1])
    selected_index = model_select.transform(selected_index.reshape((1, -1)))
    np.save(save_dir, selected_index[0])
    x_selected = model_select.transform(x_train)
    print("After feature pre-selection: {}".format(x_selected.shape))


def select_feature(x_train, y_train, x_test, feature_num=100, method="SelectKBest", alpha=0.01):
    """
    Select features based on training data(but actually we can use all data)
    :param x_train: features
    :param y_train: labels
    :param x_test: test features
    :param feature_num: feature number
    :param method: feature selection method SelectKBest or Lasso
    :return x_selected, y_train, x_test_selected: return selected feature and label
    """
    print()
    print("Selecting features using {}...".format(method))
    print("Before feature selection: {}".format(x_train.shape))
    if method == 'SelectKBest':
        from sklearn.feature_selection import f_classif
        from sklearn.feature_selection import SelectKBest
        model_select = SelectKBest(score_func=f_classif, k=feature_num)
        model_select.fit(x_train, y_train)
    else:
        clf_selec = linear_model.Lasso(alpha=alpha)
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


def train_evaluate_return_best_model(x_all, y_all, score_func=r2_score, fold_num=5, return_ave=False):
    """
    Train predefined models on data using 5-fold validation
    :param x_all: ndarray containing all features
    :param y_all: ndarray containing all labels
    :param score_func: score function
    :param fold_num: fold number to use K-fold CV
    :param return_ave: return average performance on all methods?
    :return best_model: best model trained on all data
    """
    print()
    print("Training model with K-fords...")
    kf = KFold(n_splits=fold_num, shuffle=True)
    best_score = 0
    best_idx = 0
    ave_score = 0
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
        score_mean_test /= fold_num
        score_mean_train /= fold_num
        ave_score += score_mean_test
        if not return_ave:
            print("{} \t score train: {}, score test: {}".format(model_heads[model_idx], score_mean_train, score_mean_test))
        if best_score < score_mean_test:
            best_score = score_mean_test
            best_idx = model_idx
    print("Training done")
    print("Best model: {}\t Score: {}".format(model_heads[best_idx], best_score))
    if return_ave:
        print("Average score on {} models = {}".format(len(models), ave_score/len(models)))
    best_model = models[best_idx]
    best_model.fit(x_all, y_all)
    return best_idx, best_model


def tune_model_params(x_all, y_all):
    """
    Tune models on data using 5-fold validation
    :param x_all: ndarray containing all features
    :param y_all: ndarray containing all labels
    :param score_func: score function
    :param fold_num: fold number to use K-fold CV
    :return best_model: best model trained on all data
    """
    print()
    print("Tuning model...")
    cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 400, 'max_depth': 8, 'min_child_weight': 2, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 3, 'reg_lambda': 2}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=1)
    optimized_GBM.fit(x_all, y_all)
    evalute_result = optimized_GBM.grid_scores_
    print('Result:{0}'.format(evalute_result))
    print('Best params：{0}'.format(optimized_GBM.best_params_))
    print('Best score:{0}'.format(optimized_GBM.best_score_))


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

    # configs:
    X_train_dir = './X_train.csv'
    y_train_dir = './y_train.csv'
    X_test_dir = './X_test.csv'
    gen_pre_sele_idx = False
    load_pre_filled_data = True    # if false using random forest to fill data
    pre_select_feature_num = 150
    filling_method = 'random_forest'
    selecting_method = 'Lasso'      # Lasso or SelectKBest
    lasso_alpha = 0.01
    feature_num = 120
    score_function = r2_score
    find_best_model = False     # display several preselected models' results (5-folds)
    if selecting_method == 'SelectKBest':
        y_pred_save_dir = './y_test_ensemble_{}_{}_{}.csv'.format(pre_select_feature_num, selecting_method, feature_num)
    elif selecting_method == 'Lasso':
        y_pred_save_dir = './y_test_ensemble_{}_{}_{}.csv'.format(pre_select_feature_num, selecting_method, lasso_alpha)
    else:
        y_pred_save_dir = './y_test_ensemble_{}_{}.csv'.format(pre_select_feature_num, selecting_method)
    data_x, data_y, data_x_test = load_data(x_path=X_train_dir, y_path=y_train_dir, x_test_path=X_test_dir)
    test_ID = data_x_test['id']
    if gen_pre_sele_idx:
        generate_preselect_idx(x_train_dir=X_train_dir, y_dir=y_train_dir,
                               x_test_dir=X_test_dir,
                               feature_num=pre_select_feature_num,
                               save_dir='./select_feature_idx_{}.npy'.format(pre_select_feature_num))
    x_ndarray, x_test_ndarray = fill_missing_data(data_x=data_x, data_x_test=data_x_test,
                                                  load_file=load_pre_filled_data, filling_method=filling_method,
                                                  train_file_dir="./x_filled_{}_{}.npy".format(filling_method, pre_select_feature_num),
                                                  test_file_dir='./x_test_filled_{}_{}.npy'.format(filling_method, pre_select_feature_num),
                                                  train_save_dir="./x_filled_{}_{}.npy".format(filling_method, pre_select_feature_num),
                                                  test_save_dir="./x_test_filled_{}_{}.npy".format(filling_method, pre_select_feature_num),
                                                  pre_select_idx_dir='./select_feature_idx_{}.npy'.format(pre_select_feature_num))
    y_ndarray = from_csv_to_ndarray(data=data_y)
    x_clean, y_clean = outlier_detection(x_raw=x_ndarray, y_raw=y_ndarray)
    x_select, y_select, x_test_select = select_feature(x_train=x_clean, y_train=y_clean,
                                                       x_test=x_test_ndarray, feature_num=feature_num,
                                                       method=selecting_method, alpha=lasso_alpha)

    # show correlation heat map
    train = np.concatenate((x_select, y_select.reshape((-1, 1))), axis=1)
    df = pd.DataFrame(train)
    corrmat = df.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()

    if find_best_model:
        # show some results
        _, _ = train_evaluate_return_best_model(x_all=x_select, y_all=y_select,
                                                score_func=score_function, fold_num=5)

    # =================================================
    # Ensemble + stacking
    # =================================================
    print()
    print("Ensemble start...")
    score = get_model_score(lasso, x_select, y_select)
    print("\nLasso score: {:.4f}\n".format(score))
    score = get_model_score(ENet, x_select, y_select)
    print("ElasticNet score: {:.4f}\n".format(score))
    score = get_model_score(KRR, x_select, y_select)
    print("Kernel Ridge score: {:.4f}\n".format(score))
    score = get_model_score(GBoost, x_select, y_select)
    print("Gradient Boosting score: {:.4f}\n".format(score))
    score = get_model_score(model_xgb, x_select, y_select)
    print("Xgboost score: {:.4f}\n".format(score))
    score = get_model_score(model_lgb, x_select, y_select)
    print("LGBM score: {:.4f}\n".format(score))

    # we don't use this simplified version of stacking
    # averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
    # score = get_model_score(averaged_models, x_select, y_select)
    # print("Averaged base models score: {:.4f}\n".format(score))

    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                     meta_model=lasso)
    score = get_model_score(stacked_averaged_models, x_select, y_select)
    print("Stacking Averaged models score: {:.4f}".format(score))
    stacked_averaged_models.fit(x_select, y_select)
    stacked_train_pred = stacked_averaged_models.predict(x_select)
    stacked_pred = stacked_averaged_models.predict(x_test_select)
    print(r2_score(y_select, stacked_train_pred))
    model_xgb.fit(x_select, y_select)
    xgb_train_pred = model_xgb.predict(x_select)
    xgb_pred = model_xgb.predict(x_test_select)
    print(r2_score(y_select, xgb_train_pred))
    model_lgb.fit(x_select, y_select)
    lgb_train_pred = model_lgb.predict(x_select)
    lgb_pred = model_lgb.predict(x_test_select)
    print(r2_score(y_select, lgb_train_pred))
    print('RMSLE score on train data:')
    print(r2_score(y_select, stacked_train_pred * 0.70 +
                   xgb_train_pred * 0.15 + lgb_train_pred * 0.15))
    model_ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15
    sub = pd.DataFrame()
    sub['id'] = test_ID
    sub['y'] = model_ensemble
    sub.to_csv(y_pred_save_dir, index=False)


main()
