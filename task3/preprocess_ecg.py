import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from biosppy.signals import ecg
from biosppy.plotting import plot_ecg
from QRS_utils import *
import pywt
import math
import csv
import tqdm


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


def wavelet_decomposition(sig):
    cA5, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(sig, 'bior4.4', level=5)
    coeffs = {'cA5': cA5, 'cD5': cD5, 'cD4': cD4, 'cD3': cD3, 'cD2': cD2, 'cD1': cD1}

    #plot stuff
    do_plot = False

    if do_plot:
        print('\n\n')
        print('Plot of wavelet decomposition for all levels')
        plt.subplots(figsize=(16,9))

        plt.subplot(6,1,1)
        plt.plot(coeffs['cA5'])
        plt.title('cA5')

        plt.subplot(6,1,2)
        plt.plot(coeffs['cD5'])
        plt.title('cD5')

        plt.subplot(6,1,3)
        plt.plot(coeffs['cD4'])
        plt.title('cD4')

        plt.subplot(6,1,4)
        plt.plot(coeffs['cD3'])
        plt.title('cD3')

        plt.subplot(6,1,5)
        plt.plot(coeffs['cD2'])
        plt.title('cD2')

        plt.subplot(6,1,6)
        plt.plot(coeffs['cD1'])
        plt.title('cD1')
        plt.xlabel('Index')

        plt.tight_layout()
        plt.savefig('./visualization/wavelet_decomposition.png', dpi=150)
        plt.show()

    return coeffs


def wavelet_reconstruction(coeffs, orig_data, CR, do_plot=False):
    reconstructed = pywt.waverec([coeffs['cA5'], coeffs['cD5'], coeffs['cD4'], coeffs['cD3'],
                                  coeffs['cD2'], coeffs['cD1']], 'bior4.4')

    if do_plot:
        print('\n\n')
        print('Plot of original signal through the process of compression and decompression:')
        sample_count = len(orig_data)
        sample_rate = 300
        t = np.linspace(0, sample_count / sample_rate, sample_count)
        plt.clf()
        plt.subplots(figsize=(16, 9))
        plt.plot(t, orig_data, label='Original Signal')
        plt.plot(t, reconstructed, label='Reconstructed Signal')
        plt.title('Compression Ratio: %.1f' % CR)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (mV)')
        plt.tight_layout()
        plt.legend(loc=1)
        plt.savefig('./visualization/reconstructed.png', dpi=150)
        plt.show()

    return reconstructed


def main():
    print()
    print('***************By Killer Queen***************')

    # configs:
    X_train_dir = './data/x_train.npy'
    X_test_dir = './data/x_test.npy'

    x_train = np.load(X_train_dir)
    x_test = np.load(X_test_dir)
    sample_rate = 300
    save_prefix = ['train', 'test']
    for i, set in enumerate([x_train, x_test]):
        for j in tqdm.tqdm(range(len(set))):
            x = set[j]
            x = x[~np.isnan(x)]
            out = ecg.ecg(x, sampling_rate=sample_rate, show=False)
            S_point, Q_point = QS_detect(out[1], 300, out[2], False)
            np.savez('./data/{}_{}'.format(save_prefix[i], j),
                     ts=np.array(out[0]),
                     filtered=np.array(out[1]),
                     rpeaks=np.array(out[2]),
                     templates_ts=np.array(out[3]),
                     templates=np.array(out[4]),
                     heart_rate_ts=np.array(out[5]),
                     heart_rate=np.array(out[6]),
                     s_points=S_point,
                     q_points=Q_point)

    """
    ts (array) – Signal time axis reference (seconds).
    filtered (array) – Filtered ECG signal. 
    rpeaks (array) – R-peak location indices.
    templates_ts (array) – Templates time axis reference (seconds).
    templates (array) – Extracted heartbeat templates.
    heart_rate_ts (array) – Heart rate time axis reference (seconds).
    heart_rate (array) – Instantaneous heart rate (bpm).
    s_points (array) - S location indices
    q_points (array) - Q location indices
    """


main()
