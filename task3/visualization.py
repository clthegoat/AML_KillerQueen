import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from biosppy.signals import ecg
from biosppy.plotting import plot_ecg
from hrvanalysis import get_time_domain_features
from QRS_utils import *
import pywt
import math
import csv

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
    do_plot = True

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
    y_train_dir = './data/y_train.npy'
    X_test_dir = './data/x_test.npy'

    # data_x, data_y, data_x_test = load_data(x_path=X_train_dir, y_path=y_train_dir, x_test_path=X_test_dir)
    x_train = np.load(X_train_dir)
    y_train = np.load(y_train_dir)
    x_test = np.load(X_test_dir)

    index = np.arange(len(x_train))
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]
    """
    for j in range(4):
        cnt = 0
        for i in range(len(x_train)):
            if cnt > 30:
                break
            if y_train[i] == j:
                plt.clf()
                plt.plot(x_train[i][:3000])
                plt.ylabel('class {}'.format(j))
                fig = plt.gcf()
                fig.savefig('./visualization/{}_{}_0.png'.format(j, cnt))
                cnt+=1
    """

    """
    d1 = x_train[y_train==0]
    d2 = x_train[y_train==1]
    d3 = x_train[y_train==2]
    d4 = x_train[y_train==3]
    ds = [d1, d2, d3, d4]
    std_lists = [[], [], [], []]
    for i in range(4):
        print("#Class {}: {}".format(i, len(ds[i])))
        ave_max = 0
        ave_min = 0
        ave_mean = 0
        ave_std = 0
        for sample in ds[i]:
            sample = sample[~np.isnan(sample)]
            ave_max += np.max(sample)
            ave_min += np.min(sample)
            ave_mean += np.mean(sample)
            ave_std += np.std(sample)
            std_lists[i].append(np.std(sample))
        print("ave max: {}, ave min: {}, ave mean: {}, ave std: {}".format(
            ave_max/len(ds[i]),
            ave_min / len(ds[i]),
            ave_mean / len(ds[i]),
            ave_std / len(ds[i]),
        ))
        plt.clf()
        plt.hist(np.array(std_lists[i]), bins=40)
        plt.title('class {} std'.format(i))
        plt.savefig('./visualization/class_{}_std_distribution.png'.format(i))
        plt.show()
    """
    sample_rate = 300
    cnt_max = 30
    for j in range(4):
        cnt = 1
        wavelet_energy = np.zeros((cnt_max, 6))
        for i in range(len(x_train)):
            if cnt > cnt_max:
                break
            if y_train[i] == j:

                x = x_train[i]
                x = x[~np.isnan(x)]
                sample_count = len(x)
                t = np.linspace(0, sample_count/sample_rate, sample_count)
                xFFT = np.abs(np.fft.rfft(x)/sample_count)
                xFreqs = np.linspace(0, sample_rate/2, int(sample_count/2)+1)

                plt.clf()
                plt.figure(figsize=(10, 6))
                ax0 = plt.subplot(211)
                ax0.set_xlabel('Time(s)')
                ax0.set_ylabel("Amp")
                ax0.plot(t, x)
                ax1 = plt.subplot(212)
                ax1.set_xlabel('Freq(Hz)')
                ax1.set_ylabel('Power')
                ax1.plot(xFreqs, xFFT)
                plt.title('class {}'.format(j))
                fig = plt.gcf()
                fig.savefig('./visualization/{}_{}_fft.png'.format(j, cnt))



                #b, a = butter_band_pass_filter(3, 60, sample_rate, order=4)
                #x = signal.lfilter(b, a, x)
                out = ecg.ecg(x, sampling_rate=300, show=False)
                S_pint, Q_point = QS_detect(out[1], 300, out[2], False)
                time_domain_features = get_time_domain_features(out[2])
                templates = out[4]
                plot_ecg(out[0], x, out[1], out[2], out[3], out[4], out[5], out[6], path='./visualization/{}_{}_lib.png'.format(j, cnt), show=False)
                coeffs = wavelet_decomposition(np.array(out[1]))
                a = coeffs[0]
                #for k, coeff in enumerate(coeffs):
                #    b = coeffs[coeff]
                #    b = b/np.max(np.abs(b))
                #    wavelet_energy[cnt-1][k] = np.mean(b * b)
                #_ = wavelet_reconstruction(coeffs, np.array(out[1]), 1, True)
                print(cnt)
                cnt += 1
        print('Class {}'.format(j))
        print('mean:')
        print(np.mean(wavelet_energy, axis=0))
        print('std:')
        print(np.std(wavelet_energy, axis=0))

main()
