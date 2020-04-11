import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import pywt
import tqdm
import hrvanalysis
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model


NUM_TRAIN = 5117
NUM_TEST = 3411


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


def select_feature(x_train, y_train, x_test, feature_num=300, method="SelectKBest", alpha=0.01):
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


def wavelet_decomposition(sig):
    cA5, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(sig, 'bior4.4', level=5)
    coeffs = {'cA5': cA5, 'cD5': cD5, 'cD4': cD4, 'cD3': cD3, 'cD2': cD2, 'cD1': cD1}

    return coeffs


def extract_feature(data, raw_signal):
    feature = []
    ts = np.array(data['ts'])
    filtered = np.array(data['filtered'])
    rpeaks = np.array(data['rpeaks'])
    templates_ts = np.array(data['templates_ts'])
    templates = np.array(data['templates'])
    heart_rate_ts = np.array(data['heart_rate_ts'])
    heart_rate = np.array(data['heart_rate'])
    s_points = np.array(data['s_points'])
    q_points = np.array(data['q_points'])

    feature.append(np.std(raw_signal))
    feature.append(np.std(filtered))
    # RR interval
    rr_intervals = rpeaks[1:] - rpeaks[:-1]
    feature.append(np.min(rr_intervals))
    feature.append(np.max(rr_intervals))
    feature.append(np.mean(rr_intervals))
    feature.append(np.std(rr_intervals))

    # R amplitude
    r_apt = np.abs(filtered[rpeaks])
    feature.append(np.min(r_apt))
    feature.append(np.max(r_apt))
    feature.append(np.mean(r_apt))
    feature.append(np.std(r_apt))

    # Q amplitude
    q_apt = np.abs(filtered[q_points])
    feature.append(np.min(q_apt))
    feature.append(np.max(q_apt))
    feature.append(np.mean(q_apt))
    feature.append(np.std(q_apt))

    # QRS duration
    qrs_duration = s_points - q_points
    feature.append(np.min(qrs_duration))
    feature.append(np.max(qrs_duration))
    feature.append(np.mean(qrs_duration))
    feature.append(np.std(qrs_duration))

    interpolated_nn_intervals = rr_intervals*10/3
    time_domain_feature = hrvanalysis.get_time_domain_features(interpolated_nn_intervals)
    for f_name in time_domain_feature:
        feature.append(time_domain_feature[f_name])

    # geometrical features
    geo_feature = hrvanalysis.get_geometrical_features(interpolated_nn_intervals)
    for f_name in geo_feature:
        if geo_feature[f_name] is None:
            feature.append(0)
        else:
            feature.append(geo_feature[f_name])

    # frequency domain features
    f_feature = hrvanalysis.get_frequency_domain_features(interpolated_nn_intervals)
    for f_name in f_feature:
        if f_feature[f_name] is None:
            feature.append(0)
        else:
            feature.append(f_feature[f_name])

    # csi cvi features
    csi_cvi_feature = hrvanalysis.get_csi_cvi_features(interpolated_nn_intervals)
    for f_name in csi_cvi_feature:
        if csi_cvi_feature[f_name] is None:
            feature.append(0)
        else:
            feature.append(csi_cvi_feature[f_name])

    # get_poincare_plot_features
    pp_feature = hrvanalysis.get_poincare_plot_features(interpolated_nn_intervals)
    for f_name in pp_feature:
        if pp_feature[f_name] is None:
            feature.append(0)
        else:
            feature.append(pp_feature[f_name])


    # wavelet energy
    coeffs = wavelet_decomposition(filtered)

    for k, coeff in enumerate(coeffs):
        b = coeffs[coeff]
        b = b/np.max(np.abs(b))
        feature.append(np.mean(b * b))

    # wavelet energy
    coeffs = wavelet_decomposition(raw_signal)

    for k, coeff in enumerate(coeffs):
        b = coeffs[coeff]
        b = b / np.max(np.abs(b))
        feature.append(np.mean(b * b))

    # template average
    templates_ave = np.mean(templates, axis=0)
    templates_ave = templates_ave/np.max(np.abs(templates_ave))
    for p in templates_ave:
        feature.append(p)

    # template std ave
    templates_std = np.std(templates, axis=0)
    templates_std = templates_std/np.max(np.abs(templates_std))
    for p in templates_std:
        feature.append(p)

    # wavelet 1
    wavelet_1 = coeffs['cA5']
    wavelet_1 = wavelet_1[:50]
    for p in wavelet_1:
        feature.append(p)

    return np.array(feature)


def prepare_feature(mode, save_feature=False):
    print('Preparing {} features...'.format(mode))
    if mode == 'train':
        num_sample = NUM_TRAIN
    else:
        num_sample = NUM_TEST
    if not save_feature:
        return np.load('./data/{}_features_manual.npy'.format(mode))
    raw_data = np.load('./data/x_{}.npy'.format(mode))
    features = []
    for i in tqdm.tqdm(range(num_sample)):
        data = np.load('./data/{}_{}.npz'.format(mode, i))
        raw_signal = raw_data[i]
        raw_signal = raw_signal[~np.isnan(raw_signal)]
        features.append(extract_feature(data, raw_signal))
    features = np.array(features)
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 0
    if save_feature:
        np.save('./data/{}_features_manual.npy'.format(mode), features)
    return features


def try_different_method(model, x_train, y_train, x_test, y_test, w_array):
    """
    Inner function in train_evaluate_return_best_model for model training.
    :param model: one specific model
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param w_array:
    :return score:
    """
    # x_train, y_train = over_sampling(x_train, y_train)
    """
    class_weights = [0.514373970345964, 2.931948424068768, 0.76937394247038, 7.6361940298507465]

    w_array = np.ones(y_train.shape[0], dtype='float')
    for i, val in enumerate(y_train):
        w_array[i] = class_weights[val]
    """


    if w_array is not None:
        model.fit(x_train, y_train, sample_weight=w_array)
    else:
        model.fit(x_train, y_train)
    result_test = model.predict(x_test)
    result_train = model.predict(x_train)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, result_test))
    score_test = f1_score(y_test, result_test, average='micro')
    score_train = f1_score(y_train, result_train, average='micro')
    return score_test, score_train


def evaluate_model(model, x_all, y_all, w_array=None):
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
        score_test, score_train = try_different_method(instance_model, x_train, y_train, x_test, y_test, w_array)
        print(score_test)
        print(score_train)
        score_mean_test += score_test
        score_mean_train += score_train

    score_mean_test /= n_folds
    score_mean_train /= n_folds
    print("Mean score on test set: {}".format(score_mean_test))
    print("Mean score on train set: {}".format(score_mean_train))
    if w_array is not None:
        model.fit(x_all, y_all, sample_weight=w_array)
    else:
        model.fit(x_all, y_all)
    return model, score_mean_test


def main():
    y_train = np.load('./data/y_train.npy')
    x_train = prepare_feature('train', save_feature=False)
    x_test = prepare_feature('test', save_feature=False)
    x_train, x_test = data_preprocessing(x_train, x_test)
    x_train, y_train, x_test = select_feature(x_train, y_train, x_test, feature_num=200)
    cv_params = {'C': [0.8, 1, 1.2], 'gamma': [0.0001, 0.0005, 0.001, 0.002],
                 'class_weight': [{0: 2.2, 1: 0.4444, 2: 2.4},
                                  {0: 2.6667, 1: 0.4444,
                                   2: 2.6667},
                                  {0: 2.8, 1: 0.4444, 2: 2.6666}, ]}
    other_params = {"C": 0.8, "gamma": 0.001, 'class_weight': {0: 2.6667, 1: 0.4444, 2: 2.6667}}
    # clf = SVC(class_weight='balanced')
    # clf = ensemble.RandomForestClassifier(n_estimators=200)
    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=100)
    # optimized_GBM = GridSearchCV(estimator=clf, param_grid=cv_params, scoring='balanced_accuracy', cv=5, verbose=1, n_jobs=4)
    # optimized_GBM.fit(x_train, y_train)
    # print('Best params：{0}'.format(optimized_GBM.best_params_))
    # print('Best score:{0}'.format(optimized_GBM.best_score_))

    model, score = evaluate_model(clf, x_train, y_train)
    prediction = model.predict(x_test)

    sub = pd.DataFrame()
    sub['id'] = np.arange(len(prediction))
    sub['y'] = prediction
    y_pred_save_dir = './results/y_test_xgb_{}.csv'.format(score)
    sub.to_csv(y_pred_save_dir, index=False)


main()
