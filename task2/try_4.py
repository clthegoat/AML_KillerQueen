import pandas as pd
import numpy as np
import argparse

from keras.regularizers import l2
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tqdm
import keras
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from keras import models, Sequential, optimizers
from keras import layers
from keras.layers import Dense, Activation, PReLU, BatchNormalization, Dropout
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
color = sns.color_palette()
# from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn  # ignore annoying warning (from sklearn and seaborn)


# import xgboost as xgb
# model_XGBoostClassifier = xgb.XGBClassifier()

# One Hot Encoder
def to_one_hot(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def network(feature_dimension):
    model = models.Sequential()
    model.add(Dense(64, activation='relu', input_dim=feature_dimension))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def network2(feature_dimension):
    models = Sequential()
    #W_regularizer=l2(0.0001)
    models.add(Dense(256, input_dim=feature_dimension, init='uniform', W_regularizer=l2(0.0001)))
    models.add(PReLU())
    models.add(BatchNormalization())
    models.add(Dropout(0.8))
    models.add(Dense(64, activation='relu'))
    models.add(Dropout(0.3))
    models.add(Dense(32, activation='relu'))
    models.add(Dropout(0.2))
    models.add(Dense(3, activation='softmax'))
    #models.add(Activation('softmax'))
    opt = optimizers.Adagrad(lr=0.015)
    models.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return models


def bas(y_pred, y_true):
    corrects = np.zeros(3)
    class_num = np.zeros(3)
    for i in range(len(y_pred)):
        class_num[y_true[i]] += 1
        corrects[y_true[i]] += (y_true[i] == y_pred[i])
    # BMAC = balanced_accuracy_score(y_ture, y_pred)
    BMAC = balanced_accuracy_score(y_true, y_pred)
    # return np.sum(corrects/class_num)/3
    return BMAC


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
    # PCA transform
    pca = PCA(n_components=1000)
    x_after = pca.fit_transform(x_after)
    return x_after[:len(x_raw)], x_after[len(x_raw):]

def down_sampling(x_train, y_train):
    print("Down Sampling My friend.....")
    from imblearn.under_sampling import NearMiss
    nm1 = NearMiss(version=1)
    x_train, y_train = nm1.fit_resample(x_train, y_train)
    return x_train, y_train

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
    # Using SMOTE: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
    # an Over-sampling approach
    # Over sampling on training and validation data
    # sm = SMOTE(sampling_strategy='auto', random_state=10)
    # sm = SVMSMOTE(random_state=0)
    sm = SMOTEENN(random_state=0)
    # sm = SMOTETomek(ratio='auto')
    x_train, y_train = sm.fit_resample(x_train, y_train)

    # x_train, y_train = sm.fit_resample(x_train, y_train)
    # X_train, X_val, y_train, y_val = train_test_split(X_train,y,test_size=0.2,random_state=7)
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


def select_feature(x_train, y_train, x_test, feature_num=900, method="SelectKBest", alpha=0.01):
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


# def select_feature(x_train, y_train, x_test):
#     """
#     Select features based on training data(but actually we can use all data)
#     :param x_train: features
#     :param y_train: labels
#     :param x_test: test features
#     :return x_selected, y_train, x_test_selected: return selected feature and label
#     """
#     print()
#     print("Selecting features...")
#     print("Before feature selection: {}".format(x_train.shape))
#     x_selected = x_train
#     x_test_selected = x_test
#     print("After feature selection: {}".format(x_selected.shape))
#     return x_selected, y_train, x_test_selected


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
    boolean = lambda x: bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='nn', help="choose models options:nn, xgboost, LogisticRegression")
    parser.add_argument('--Is_oversampling', type=boolean, default='False', help="If over sampling")
    parser.add_argument('--Is_downsampling', type=boolean, default='True', help="If down sampling")
    opt = parser.parse_args()
    # configs:
    X_train_dir = './data/X_train.csv'
    y_train_dir = './data/y_train.csv'
    X_test_dir = './data/X_test.csv'
    y_pred_save_dir = './y_test_try.csv'
    data_x, data_y, data_x_test = load_data(x_path=X_train_dir, y_path=y_train_dir, x_test_path=X_test_dir)
    test_ID = data_x_test['id']
    y_train = from_csv_to_ndarray(data=data_y)
    x_train = from_csv_to_ndarray(data=data_x)
    x_test = from_csv_to_ndarray(data=data_x_test)
    x_train, x_test = data_preprocessing(x_train, x_test)  # Normalizaiton and ...
    # Split the data before oversampling
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=7)
    if opt.Is_oversampling:
        x_train_all, y_train_all = over_sampling(X_train, y_train)
    elif opt.Is_downsampling:
        x_train_all, y_train_all = down_sampling(X_train, y_train)
    else:
        x_train_all, y_train_all = X_train, y_train
    # x_train, y_train, x_test = select_feature(x_train, y_train, x_test)
    # x_train, y_train, x_test = select_feature(x_train, y_train, x_test, feature_num=950, method="SelectKBest", alpha=0.01)
    # Possible Options: LogisticRegression, XGBoost, neurual network....
    if opt.method == 'nn':
        #X_train, X_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.1, random_state=7)
        one_hot_labels = keras.utils.to_categorical(y_train_all, num_classes=3)
        one_hot_label_val = keras.utils.to_categorical(y_val, num_classes=3)
        print(one_hot_labels)
        print(x_train_all.shape)
        print(y_train_all.shape)
        model = network2(x_train_all.shape[1])
        class_weights = {0:5,
                         1:1.,
                         2:5}
        model.fit(x_train_all, one_hot_labels, epochs=30, batch_size=256, class_weight=class_weights,validation_data=(X_val, one_hot_label_val))
        prediction = model.predict_classes(x_test)
        y_pred = model.predict_classes(x_train_all)
        BMAC_all = balanced_accuracy_score(y_train_all, y_pred)
        print("BMAC on ALL training data", BMAC_all)
        y_val_pred = model.predict_classes(X_val)
        BMAC_val = balanced_accuracy_score(y_val_pred, y_val)
        print("BMAC on validation data", BMAC_val)
    else:
        if opt.method == 'LogisticRegression':
            chosen_model = LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced')
        elif opt.method == 'xgboost':
            chosen_model = model_XGBoostClassifier
        model = evaluate_model(chosen_model, x_train, y_train)
        prediction = model.predict(x_test)
    # Neural Network Approach

    # show correlation heat map
    """
    train = np.concatenate((x_select, y_select.reshape((-1, 1))), axis=1)
    df = pd.DataFrame(train)
    corrmat = df.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()
    """

    sub = pd.DataFrame()
    sub['id'] = test_ID
    sub['y'] = prediction
    sub.to_csv(y_pred_save_dir, index=False)


main()