import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal
from biosppy.signals import ecg
from biosppy.plotting import plot_ecg
from hrvanalysis import get_time_domain_features
import math
import csv


def plot_some_imgs(train_eeg1, train_eeg2, train_emg, y_train):
    for j in range(1, 4):
        cnt = 0
        for i in range(len(train_eeg1)):
            if cnt > 10:
                break
            if y_train[i] == j:
                t = np.linspace(0, 512 / 128, 512)
                plt.clf()
                plt.figure(figsize=(10, 6))
                ax0 = plt.subplot(311)
                ax0.set_xlabel('Time(s), class: {}'.format(j))
                ax0.set_ylabel("eeg_1 Amp")
                ax0.plot(t, train_eeg1[i])
                ax1 = plt.subplot(312)
                ax1.set_xlabel('Time(s), class: {}'.format(j))
                ax1.set_ylabel("eeg_2 Amp")
                ax1.plot(t, train_eeg2[i])
                ax2 = plt.subplot(313)
                ax2.set_xlabel('Time(s), class: {}'.format(j))
                ax2.set_ylabel("emg Amp")
                ax2.plot(t, train_emg[i])
                fig = plt.gcf()
                fig.savefig('./visualization/{}_{}.png'.format(j, cnt))
                cnt += 1
    return


def plot_full_length_img(train_eeg1, train_eeg2, train_emg, y_train):
    size = 4096*2
    sub1_eeg_1 = train_eeg1[:size, :].flatten()
    sub1_eeg_2 = train_eeg2[:size, :].flatten()
    sub1_emg = train_emg[:size, :].flatten()
    sub1_y = np.repeat(y_train[:size].reshape((size, 1)), 512, axis=1).flatten()
    sub2_eeg_1 = train_eeg1[21600:21600+size, :].flatten()
    sub2_eeg_2 = train_eeg2[21600:21600+size, :].flatten()
    sub2_emg = train_emg[21600:21600+size, :].flatten()
    sub2_y = np.repeat(y_train[21600:21600+size].reshape((size, 1)), 512, axis=1).flatten()
    sub3_eeg_1 = train_eeg1[2 * 21600:2 * 21600+size, :].flatten()
    sub3_eeg_2 = train_eeg2[2 * 21600:2 * 21600+size, :].flatten()
    sub3_emg = train_emg[2 * 21600:2 * 21600+size, :].flatten()
    sub3_y = np.repeat(y_train[2 * 21600:2 * 21600+size].reshape((size, 1)), 512, axis=1).flatten()
    eeg_1 = [sub1_eeg_1, sub2_eeg_1, sub3_eeg_1]
    eeg_2 = [sub1_eeg_2, sub2_eeg_2, sub3_eeg_2]
    emg = [sub1_emg, sub2_emg, sub3_emg]
    y = [sub1_y, sub2_y, sub3_y]
    for i in range(3):
        t = np.linspace(0, size*512 / 128, size*512)
        plt.clf()
        plt.figure(figsize=(320, 8))
        ax0 = plt.subplot(411)
        ax0.set_xlabel('Time(s), sub: {}'.format(i))
        ax0.set_ylabel("eeg_1 Amp")
        ax0.plot(t, eeg_1[i])
        ax1 = plt.subplot(412)
        ax1.set_xlabel('Time(s), sub: {}'.format(i))
        ax1.set_ylabel("eeg_2 Amp")
        ax1.plot(t, eeg_2[i])
        ax2 = plt.subplot(413)
        ax2.set_xlabel('Time(s), sub: {}'.format(i))
        ax2.set_ylabel("emg Amp")
        ax2.plot(t, emg[i])
        ax3 = plt.subplot(414)
        ax3.set_xlabel('Time(s), sub: {}'.format(i))
        ax3.set_ylabel("class")
        ax3.plot(t, y[i])
        fig = plt.gcf()
        fig.savefig('./visualization/sub_{}_all.png'.format(i))


def print_statistics(train_eeg1, train_eeg2, train_emg, y_train):
    names = ['eeg1', 'eeg2', 'emg']
    d1 = [train_eeg1[y_train == 1], train_eeg2[y_train == 1], train_emg[y_train == 1]]
    d2 = [train_eeg1[y_train == 2], train_eeg2[y_train == 2], train_emg[y_train == 2]]
    d3 = [train_eeg1[y_train == 3], train_eeg2[y_train == 3], train_emg[y_train == 3]]
    ds = [d1, d2, d3]
    stat_list = [[[[], [], []], [[], [], []], [[], [], []]],
                 [[[], [], []], [[], [], []], [[], [], []]],
                 [[[], [], []], [[], [], []], [[], [], []]]]
    for j, d in enumerate(ds):
        print("#Class {}: {}".format(j, len(d[0])))
        for i, name in enumerate(names):
            ave_max = 0
            ave_min = 0
            ave_mean = 0
            ave_std = 0
            ave_eng = 0
            for sample in d[i]:
                sample = sample[~np.isnan(sample)]
                ave_max += np.max(sample)
                ave_min += np.min(sample)
                ave_mean += np.mean(sample)
                ave_std += np.std(sample)
                ave_eng += np.sum(sample*sample)
                stat_list[j][i][0].append(np.max(sample))
                stat_list[j][i][1].append(np.std(sample))
                stat_list[j][i][2].append(np.sum(sample*sample))

            print("{} ave max: {}, ave min: {}, ave mean: {}, ave std: {}, ave energy: {}".format(
                name,
                ave_max / len(d[i]),
                ave_min / len(d[i]),
                ave_mean / len(d[i]),
                ave_std / len(d[i]),
                ave_eng / len(d[i]),
            ))
    for idx, name in enumerate(names):
        eeg_ranges = [(0, 0.0035), (0, 0.001), (0, 0.0005)]
        emg_ranges = [(0, 0.003), (0, 0.005), (0, 0.00025)]
        cnt = 1
        plt.figure(figsize=(20, 15))
        if name == 'emg':
            ranges = emg_ranges
        else:
            ranges = eeg_ranges
        for class_num in range(1, 4):
            stat_names = ['max', 'std', 'energy']
            for i, stat_name in enumerate(stat_names):
                ax0 = plt.subplot(3, 3, cnt)
                ax0.set_xlabel('class {}, {} {}'.format(class_num, name, stat_name))
                ax0.hist(np.array(stat_list[class_num-1][idx][i]), bins=80, range=ranges[i])
                cnt += 1
        fig = plt.gcf()
        fig.savefig('./visualization/distribution_{}.png'.format(name))

    return


def shuffle(train_eeg1, train_eeg2, train_emg, y_train):
    index = np.arange(len(train_eeg1))
    np.random.shuffle(index)
    train_eeg1 = train_eeg1[index]
    train_eeg2 = train_eeg2[index]
    train_emg = train_emg[index]
    y_train = y_train[index]
    return train_eeg1, train_eeg2, train_emg, y_train


def plot_fft(train_eeg1, train_eeg2, train_emg, y_train):
    sample_rate = 128
    cnt_max = 10
    for j in range(1, 4):
        cnt = 1
        for i in range(len(train_eeg1)):
            if cnt > cnt_max:
                break
            if y_train[i] == j:
                eeg_1 = train_eeg1[i]
                eeg_2 = train_eeg2[i]
                emg = train_emg[i]
                label = y_train[i]
                sample_count = 512
                plt.clf()
                plt.figure(figsize=(30, 6))
                names = ['eeg1', 'eeg2', 'emg']
                for k, sig in enumerate([eeg_1, eeg_2, emg]):
                    t = np.linspace(0, sample_count / sample_rate, sample_count)
                    xFFT = np.abs(np.fft.rfft(sig) / sample_count)
                    xFreqs = np.linspace(0, sample_rate / 2, int(sample_count / 2) + 1)

                    ax0 = plt.subplot(2, 3, k+1)
                    ax0.set_xlabel('Time(s) Class: {} {}'.format(j, names[k]))
                    ax0.set_ylabel("Amp")
                    ax0.plot(t, sig)
                    ax1 = plt.subplot(2, 3, k+4)
                    ax1.set_xlabel('Freq(Hz)')
                    ax1.set_ylabel('Power')
                    ax1.plot(xFreqs, xFFT)
                    plt.title('class {}'.format(j))
                fig = plt.gcf()
                fig.savefig('./visualization/{}_{}_fft.png'.format(j, cnt))
                cnt+=1

    return


def main():
    print()
    print('***************By Killer Queen***************')

    train_eeg1 = np.load('./data/train_eeg1.npy')
    train_eeg2 = np.load('./data/train_eeg2.npy')
    train_emg = np.load('./data/train_emg.npy')
    y_train = np.load('./data/train_labels.npy')
    if not os.path.exists('./visualization'):
        os.mkdir('./visualization')

    # train_eeg1, train_eeg2, train_emg, y_train = shuffle(train_eeg1, train_eeg2, train_emg, y_train)
    # plot_some_imgs(train_eeg1, train_eeg2, train_emg, y_train)
    # plot_full_length_img(train_eeg1, train_eeg2, train_emg, y_train)
    # print_statistics(train_eeg1, train_eeg2, train_emg, y_train)
    plot_fft(train_eeg1, train_eeg2, train_emg, y_train)


main()
