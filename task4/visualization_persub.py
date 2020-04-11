import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy import signal
import math
import csv
import seaborn as sns


def generate_per_sub_features():
    # generate features npy files with per sub normalization
    mode = 'train'

    features = []
    features_sub0_eeg1 = []
    features_sub0_eeg2 = []
    features_sub1_eeg1 = []
    features_sub1_eeg2 = []
    features_sub2_eeg1 = []
    features_sub2_eeg2 = []
    if mode == 'train':
        eeg1_before = np.load('./data/train_eeg1_fft_features_new.npy')
        eeg2_before = np.load('./data/train_eeg2_fft_features_new.npy')
        emg_before = np.load('./data/train_emg_fft_features_new.npy')
    else:
        eeg1_before = np.load('./data/test_eeg1_fft_features_new.npy')
        eeg2_before = np.load('./data/test_eeg2_fft_features_new.npy')
        emg_before = np.load('./data/test_emg_fft_features_new.npy')

    eeg1 = np.log(eeg1_before)
    eeg2 = np.log(eeg2_before)
    emg = np.log(emg_before)

    # print(eeg1.shape)
    # print(eeg2.shape)
    # print(emg.shape)

    eeg1_sub0 = eeg1[:21600, :, :]
    eeg1_sub1 = eeg1[21600:2 * 21600, :, :]
    eeg1_sub2 = eeg1[2 * 21600:, :, :]
    print('eeg1_sub0', eeg1_sub0.shape)
    print('eeg1_sub1', eeg1_sub1.shape)
    print('eeg1_sub2', eeg1_sub2.shape)

    eeg2_sub0 = eeg2[:21600, :, :]
    eeg2_sub1 = eeg2[21600:2 * 21600, :, :]
    eeg2_sub2 = eeg2[2 * 21600:, :, :]

    emg_sub0 = emg[:21600, :, :]
    emg_sub1 = emg[21600:2 * 21600, :, :]
    emg_sub2 = emg[2 * 21600:, :, :]

    eeg1_sub0_reshape = eeg1_sub0.reshape(48, 32 * 21600)
    eeg1_sub1_reshape = eeg1_sub1.reshape(48, 32 * 21600)
    eeg1_sub2_reshape = eeg1_sub2.reshape(48, 32 * 21600)

    eeg1_sub0_reshape_mean = np.mean(eeg1_sub0_reshape, axis=1, keepdims=True)
    eeg1_sub0_reshape_std = np.std(eeg1_sub0_reshape, axis=1, keepdims=True)
    eeg1_sub1_reshape_mean = np.mean(eeg1_sub1_reshape, axis=1, keepdims=True)
    eeg1_sub1_reshape_std = np.std(eeg1_sub1_reshape, axis=1, keepdims=True)
    eeg1_sub2_reshape_mean = np.mean(eeg1_sub2_reshape, axis=1, keepdims=True)
    eeg1_sub2_reshape_std = np.std(eeg1_sub2_reshape, axis=1, keepdims=True)

    eeg2_sub0_reshape = eeg2_sub0.reshape(48, 32 * 21600)
    eeg2_sub1_reshape = eeg2_sub1.reshape(48, 32 * 21600)
    eeg2_sub2_reshape = eeg2_sub2.reshape(48, 32 * 21600)

    eeg2_sub0_reshape_mean = np.mean(eeg2_sub0_reshape, axis=1, keepdims=True)
    eeg2_sub0_reshape_std = np.std(eeg2_sub0_reshape, axis=1, keepdims=True)
    eeg2_sub1_reshape_mean = np.mean(eeg2_sub1_reshape, axis=1, keepdims=True)
    eeg2_sub1_reshape_std = np.std(eeg2_sub1_reshape, axis=1, keepdims=True)
    eeg2_sub2_reshape_mean = np.mean(eeg2_sub2_reshape, axis=1, keepdims=True)
    eeg2_sub2_reshape_std = np.std(eeg2_sub2_reshape, axis=1, keepdims=True)

    emg_sub0_reshape = emg_sub0.reshape(60, 32 * 21600)
    emg_sub1_reshape = emg_sub1.reshape(60, 32 * 21600)
    emg_sub2_reshape = emg_sub2.reshape(60, 32 * 21600)

    for eeg1_sub0_sig, eeg2_sub0_sig, emg_sub0_sig in tqdm(zip(eeg1_sub0, eeg2_sub0, emg_sub0)):
        # print('eeg1_sub0_sig', eeg1_sub0_sig.shape)
        eeg1_sub0_sig = (eeg1_sub0_sig - eeg1_sub0_reshape_mean) / eeg1_sub0_reshape_std
        # print('eeg1_sub0_sig', eeg1_sub0_sig.shape)
        # print('np.mean', np.mean(eeg2_sub0_reshape, axis=1, keepdims=True).shape)
        eeg2_sub0_sig = (eeg2_sub0_sig - eeg2_sub0_reshape_mean) / eeg2_sub0_reshape_std
        # print('eeg2_sub0_sig', eeg2_sub0_sig.shape)

        emg_sub0_sig = np.sum(emg_sub0_sig, axis=0, keepdims=True)
        # print('emg_sig', emg_sub0_sig.shape)
        emg_sub0_sig = emg_sub0_sig.repeat(eeg1_sub0_sig.shape[0], axis=0)
        # print('emg_sig_after', emg_sub0_sig.shape)
        eeg1_sub0_sig = eeg1_sub0_sig.reshape((eeg1_sub0_sig.shape[0], eeg1_sub0_sig.shape[1]))
        eeg2_sub0_sig = eeg2_sub0_sig.reshape((eeg1_sub0_sig.shape[0], eeg1_sub0_sig.shape[1]))
        emg_sub0_sig = emg_sub0_sig.reshape((eeg1_sub0_sig.shape[0], eeg1_sub0_sig.shape[1]))
        # print(eeg1_sub0_sig.shape)
        # print(eeg2_sub0_sig.shape)
        # print(emg_sub0_sig.shape)
        features_sub0_eeg1.append(eeg1_sub0_sig)
        features_sub0_eeg2.append(eeg2_sub0_sig)
        # features.append(np.concatenate((eeg1_sub0_sig, eeg2_sub0_sig, emg_sub0_sig), axis=2))

    for eeg1_sub1_sig, eeg2_sub1_sig, emg_sub1_sig in tqdm(zip(eeg1_sub1, eeg2_sub1, emg_sub1)):
        # print('eeg1_sub1_sig', eeg1_sub1_sig.shape)
        eeg1_sub1_sig = (eeg1_sub1_sig - eeg1_sub1_reshape_mean) / eeg1_sub1_reshape_std
        # print('eeg1_sub1_sig', eeg1_sub1_sig.shape)
        eeg2_sub1_sig = (eeg2_sub1_sig - eeg2_sub1_reshape_mean) / eeg2_sub1_reshape_std
        # print('eeg2_sub1_sig', eeg2_sub1_sig.shape)

        emg_sub1_sig = np.sum(emg_sub1_sig, axis=0, keepdims=True)
        # print('emg_sig', emg_sub1_sig.shape)
        emg_sub1_sig = emg_sub1_sig.repeat(eeg1_sub1_sig.shape[0], axis=0)
        # print('emg_sig_after', emg_sub1_sig.shape)
        eeg1_sub1_sig = eeg1_sub1_sig.reshape((eeg1_sub1_sig.shape[0], eeg1_sub1_sig.shape[1]))
        eeg2_sub1_sig = eeg2_sub1_sig.reshape((eeg1_sub1_sig.shape[0], eeg1_sub1_sig.shape[1]))
        emg_sub1_sig = emg_sub1_sig.reshape((eeg1_sub1_sig.shape[0], eeg1_sub1_sig.shape[1]))
        # print(eeg1_sub1_sig.shape)
        # print(eeg2_sub1_sig.shape)
        # print(emg_sub1_sig.shape)
        features_sub1_eeg1.append(eeg1_sub1_sig)
        features_sub1_eeg2.append(eeg2_sub1_sig)
        # features.append(np.concatenate((eeg1_sub1_sig, eeg2_sub1_sig, emg_sub1_sig), axis=2))

    for eeg1_sub2_sig, eeg2_sub2_sig, emg_sub2_sig in tqdm(zip(eeg1_sub2, eeg2_sub2, emg_sub2)):
        # print('eeg1_sub2_sig', eeg1_sub2_sig.shape)
        eeg1_sub2_sig = (eeg1_sub2_sig - eeg1_sub2_reshape_mean) / eeg1_sub2_reshape_std
        # print('eeg1_sub2_sig', eeg1_sub2_sig.shape)
        eeg2_sub2_sig = (eeg2_sub2_sig - eeg2_sub2_reshape_mean) / eeg2_sub2_reshape_std
        # print('eeg2_sub2_sig', eeg2_sub2_sig.shape)

        emg_sub2_sig = np.sum(emg_sub2_sig, axis=0, keepdims=True)
        # print('emg_sig', emg_sub2_sig.shape)
        emg_sub2_sig = emg_sub2_sig.repeat(eeg1_sub2_sig.shape[0], axis=0)
        # print('emg_sig_after', emg_sub2_sig.shape)
        eeg1_sub2_sig = eeg1_sub2_sig.reshape((eeg1_sub2_sig.shape[0], eeg1_sub2_sig.shape[1]))
        eeg2_sub2_sig = eeg2_sub2_sig.reshape((eeg1_sub2_sig.shape[0], eeg1_sub2_sig.shape[1]))
        emg_sub2_sig = emg_sub2_sig.reshape((eeg1_sub2_sig.shape[0], eeg1_sub2_sig.shape[1]))
        # print(eeg1_sub2_sig.shape)
        # print(eeg2_sub2_sig.shape)
        # print(emg_sub2_sig.shape)
        features_sub2_eeg1.append(eeg1_sub2_sig)
        features_sub2_eeg2.append(eeg2_sub2_sig)
        # features.append(np.concatenate((eeg1_sub2_sig, eeg2_sub2_sig, emg_sub2_sig), axis=2))

    features_sub0_eeg1 = np.array(features_sub0_eeg1)
    features_sub0_eeg2 = np.array(features_sub0_eeg2)
    features_sub1_eeg1 = np.array(features_sub1_eeg1)
    features_sub1_eeg2 = np.array(features_sub1_eeg2)
    features_sub2_eeg1 = np.array(features_sub2_eeg1)
    features_sub2_eeg2 = np.array(features_sub2_eeg2)
    # print(features_sub0_eeg1.shape)
    np.save('./data/features_sub0_eeg1.npy', np.array(features_sub0_eeg1))
    np.save('./data/features_sub0_eeg2.npy', np.array(features_sub0_eeg2))
    np.save('./data/features_sub1_eeg1.npy', np.array(features_sub1_eeg1))
    np.save('./data/features_sub1_eeg2.npy', np.array(features_sub1_eeg2))
    np.save('./data/features_sub2_eeg1.npy', np.array(features_sub2_eeg1))
    np.save('./data/features_sub2_eeg2.npy', np.array(features_sub2_eeg2))


def generate_without_per_sub_features():
    # generate features npy files without per sub normalization
    mode = 'train'
    if mode == 'train':
        eeg1_before = np.load('./data/train_eeg1_fft_features_new.npy')
        eeg2_before = np.load('./data/train_eeg2_fft_features_new.npy')
        emg_before = np.load('./data/train_emg_fft_features_new.npy')
    else:
        eeg1_before = np.load('./data/test_eeg1_fft_features_new.npy')
        eeg2_before = np.load('./data/test_eeg2_fft_features_new.npy')
        emg_before = np.load('./data/test_emg_fft_features_new.npy')

    eeg1 = np.log(eeg1_before)
    eeg2 = np.log(eeg2_before)
    emg = np.log(emg_before)
    # print('eeg1.shape', eeg1.shape)
    # print('eeg2.shape', eeg2.shape)
    # print('emg.shape', emg.shape)
    features_eeg1 = []
    features_eeg2 = []
    for eeg1_sig, eeg2_sig, emg_sig in tqdm(zip(eeg1, eeg2, emg)):
        eeg1_sig = (eeg1_sig - np.mean(eeg1_sig, axis=1, keepdims=True)) / np.std(eeg1_sig, axis=1, keepdims=True)
        # print('eeg1_sig', eeg1_sig.shape) # eeg1_sig (48, 32)
        eeg2_sig = (eeg2_sig - np.mean(eeg2_sig, axis=1, keepdims=True)) / np.std(eeg2_sig, axis=1, keepdims=True)
        # print('eeg2_sig', eeg2_sig.shape) # eeg2_sig (48, 32)
        emg_sig = np.sum(emg_sig, axis=0, keepdims=True)
        # print('emg_sig', emg_sig.shape) # emg_sig (1, 32)
        emg_sig = emg_sig.repeat(eeg1_sig.shape[0], axis=0)
        # print('emg_sig_after', emg_sig.shape) # emg_sig_after (48, 32)
        eeg1_sig = eeg1_sig.reshape((eeg1_sig.shape[0], eeg1_sig.shape[1]))
        eeg2_sig = eeg2_sig.reshape((eeg1_sig.shape[0], eeg1_sig.shape[1]))
        emg_sig = emg_sig.reshape((eeg1_sig.shape[0], eeg1_sig.shape[1]))
        # eeg1_sig = np.log(eeg1_sig)
        # eeg2_sig = np.log(eeg2_sig)
        # emg_sig = np.log(emg_sig)
        # print(eeg1_sig.shape)
        # print(eeg2_sig.shape)
        # print(emg_sig.shape)
        features_eeg1.append(eeg1_sig)
        features_eeg2.append(eeg2_sig)

    features_eeg1 = np.array(features_eeg1)
    features_eeg2 = np.array(features_eeg2)
    # print(features_eeg1.shape)

    features_sub0_eeg1 = features_eeg1[:21600, :, :]
    features_sub0_eeg2 = features_eeg2[:21600, :, :]
    features_sub1_eeg1 = features_eeg1[21600:2 * 21600, :, :]
    features_sub1_eeg2 = features_eeg2[21600:2 * 21600, :, :]
    features_sub2_eeg1 = features_eeg1[2 * 21600:, :, :]
    features_sub2_eeg2 = features_eeg2[2 * 21600:, :, :]

    features_sub0_eeg1 = np.array(features_sub0_eeg1)
    features_sub0_eeg2 = np.array(features_sub0_eeg2)
    features_sub1_eeg1 = np.array(features_sub1_eeg1)
    features_sub1_eeg2 = np.array(features_sub1_eeg2)
    features_sub2_eeg1 = np.array(features_sub2_eeg1)
    features_sub2_eeg2 = np.array(features_sub2_eeg2)

    # print(features_sub0_eeg1.shape)
    np.save('./data/features_sub0_eeg1_no_pb.npy', np.array(features_sub0_eeg1))
    np.save('./data/features_sub0_eeg2_no_pb.npy', np.array(features_sub0_eeg2))
    np.save('./data/features_sub1_eeg1_no_pb.npy', np.array(features_sub1_eeg1))
    np.save('./data/features_sub1_eeg2_no_pb.npy', np.array(features_sub1_eeg2))
    np.save('./data/features_sub2_eeg1_no_pb.npy', np.array(features_sub2_eeg1))
    np.save('./data/features_sub2_eeg2_no_pb.npy', np.array(features_sub2_eeg2))


def gen_PSD_data(persub):
    if persub: 
    # features with per subject normalization
        features_sub0_eeg1 = np.load('./data/features_sub0_eeg1.npy')
        features_sub0_eeg2 = np.load('./data/features_sub0_eeg2.npy')
        features_sub1_eeg1 = np.load('./data/features_sub1_eeg1.npy')
        features_sub1_eeg2 = np.load('./data/features_sub1_eeg2.npy')
        features_sub2_eeg1 = np.load('./data/features_sub2_eeg1.npy')
        features_sub2_eeg2 = np.load('./data/features_sub2_eeg2.npy')
    elif persub == False:
    # features without per subject normalization
        features_sub0_eeg1 = np.load('./data/features_sub0_eeg1_no_pb.npy')
        features_sub0_eeg2 = np.load('./data/features_sub0_eeg2_no_pb.npy')
        features_sub1_eeg1 = np.load('./data/features_sub1_eeg1_no_pb.npy')
        features_sub1_eeg2 = np.load('./data/features_sub1_eeg2_no_pb.npy')
        features_sub2_eeg1 = np.load('./data/features_sub2_eeg1_no_pb.npy')
        features_sub2_eeg2 = np.load('./data/features_sub2_eeg2_no_pb.npy')

    y_train = np.load('./data/train_labels.npy')
    y_train_sub0 = y_train[:21600]
    y_train_sub1 = y_train[21600:2 * 21600]
    y_train_sub2 = y_train[2 * 21600:]

    # c0 means class1: WAKE phase, c1 means class2: NREM phase, c3 means class3: REM phase.
    idx_sub0_c0 = np.where(y_train_sub0 == 1)[0]
    n_sub0_c0 = len(idx_sub0_c0)
    idx_sub0_c1 = np.where(y_train_sub0 == 2)[0]
    n_sub0_c1 = len(idx_sub0_c1)
    idx_sub0_c2 = np.where(y_train_sub0 == 3)[0]
    n_sub0_c2 = len(idx_sub0_c2)
    idx_sub1_c0 = np.where(y_train_sub1 == 1)[0]
    n_sub1_c0 = len(idx_sub1_c0)
    idx_sub1_c1 = np.where(y_train_sub1 == 2)[0]
    n_sub1_c1 = len(idx_sub1_c1)
    idx_sub1_c2 = np.where(y_train_sub1 == 3)[0]
    n_sub1_c2 = len(idx_sub1_c2)
    idx_sub2_c0 = np.where(y_train_sub2 == 1)[0]
    n_sub2_c0 = len(idx_sub2_c0)
    idx_sub2_c1 = np.where(y_train_sub2 == 2)[0]
    n_sub2_c1 = len(idx_sub2_c1)
    idx_sub2_c2 = np.where(y_train_sub2 == 3)[0]
    n_sub2_c2 = len(idx_sub2_c2)

    features_sub0_eeg1_c0 = features_sub0_eeg1[idx_sub0_c0]
    features_sub0_eeg1_c1 = features_sub0_eeg1[idx_sub0_c1]
    features_sub0_eeg1_c2 = features_sub0_eeg1[idx_sub0_c2]
    features_sub1_eeg1_c0 = features_sub1_eeg1[idx_sub1_c0]
    features_sub1_eeg1_c1 = features_sub1_eeg1[idx_sub1_c1]
    features_sub1_eeg1_c2 = features_sub1_eeg1[idx_sub1_c2]
    features_sub2_eeg1_c0 = features_sub2_eeg1[idx_sub2_c0]
    features_sub2_eeg1_c1 = features_sub2_eeg1[idx_sub2_c1]
    features_sub2_eeg1_c2 = features_sub2_eeg1[idx_sub2_c2]

    features_sub0_eeg2_c0 = features_sub0_eeg2[idx_sub0_c0]
    features_sub0_eeg2_c1 = features_sub0_eeg2[idx_sub0_c1]
    features_sub0_eeg2_c2 = features_sub0_eeg2[idx_sub0_c2]
    features_sub1_eeg2_c0 = features_sub1_eeg2[idx_sub1_c0]
    features_sub1_eeg2_c1 = features_sub1_eeg2[idx_sub1_c1]
    features_sub1_eeg2_c2 = features_sub1_eeg2[idx_sub1_c2]
    features_sub2_eeg2_c0 = features_sub2_eeg2[idx_sub2_c0]
    features_sub2_eeg2_c1 = features_sub2_eeg2[idx_sub2_c1]
    features_sub2_eeg2_c2 = features_sub2_eeg2[idx_sub2_c2]

    # num and t are both time axes, f is frequency axis.
    # so we reshape it into (f, t * num) first,
    # then reshape into (f*t*num,) with f preserved:
    # i.e. each element corresponding f from 0 - 24, then repeat again
    [num, f, t] = features_sub0_eeg1_c0.shape
    print('features_sub0_eeg1/eeg2_c0.shape', features_sub0_eeg1_c0.shape)
    features_sub0_eeg1_c0_reshape = features_sub0_eeg1_c0.reshape(f, t * num)
    features_sub0_eeg2_c0_reshape = features_sub0_eeg2_c0.reshape(f, t * num)
    print('features_sub0_eeg1/eeg2_c0_reshape', features_sub0_eeg1_c0_reshape.shape)
    features_sub0_eeg1_c0_reshape_2 = np.transpose(features_sub0_eeg1_c0_reshape).reshape(-1)
    features_sub0_eeg2_c0_reshape_2 = np.transpose(features_sub0_eeg2_c0_reshape).reshape(-1)
    print('features_sub0_eeg1/eeg2_c0_reshape_2', features_sub0_eeg2_c0_reshape_2.shape)

    [num, f, t] = features_sub0_eeg1_c1.shape
    print('features_sub0_eeg1/eeg2_c1.shape', features_sub0_eeg1_c1.shape)
    features_sub0_eeg1_c1_reshape = features_sub0_eeg1_c1.reshape(f, t * num)
    features_sub0_eeg2_c1_reshape = features_sub0_eeg2_c1.reshape(f, t * num)
    print('features_sub0_eeg1/eeg2_c1_reshape', features_sub0_eeg1_c1_reshape.shape)
    features_sub0_eeg1_c1_reshape_2 = np.transpose(features_sub0_eeg1_c1_reshape).reshape(-1)
    features_sub0_eeg2_c1_reshape_2 = np.transpose(features_sub0_eeg2_c1_reshape).reshape(-1)
    print('features_sub0_eeg1/eeg2_c1_reshape_2', features_sub0_eeg2_c1_reshape_2.shape)

    [num, f, t] = features_sub0_eeg1_c2.shape
    print('features_sub0_eeg1/eeg2_c2.shape', features_sub0_eeg1_c2.shape)
    features_sub0_eeg1_c2_reshape = features_sub0_eeg1_c2.reshape(f, t * num)
    features_sub0_eeg2_c2_reshape = features_sub0_eeg2_c2.reshape(f, t * num)
    print('features_sub0_eeg1/eeg2_c2_reshape', features_sub0_eeg1_c2_reshape.shape)
    features_sub0_eeg1_c2_reshape_2 = np.transpose(features_sub0_eeg1_c2_reshape).reshape(-1)
    features_sub0_eeg2_c2_reshape_2 = np.transpose(features_sub0_eeg2_c2_reshape).reshape(-1)
    print('features_sub0_eeg1/eeg2_c2_reshape_2', features_sub0_eeg2_c2_reshape_2.shape)

    [num, f, t] = features_sub1_eeg1_c0.shape
    print('features_sub1_eeg1/eeg2_c0.shape', features_sub1_eeg1_c0.shape)
    features_sub1_eeg1_c0_reshape = features_sub1_eeg1_c0.reshape(f, t * num)
    features_sub1_eeg2_c0_reshape = features_sub1_eeg2_c0.reshape(f, t * num)
    print('features_sub1_eeg1/eeg2_c0_reshape', features_sub1_eeg1_c0_reshape.shape)
    features_sub1_eeg1_c0_reshape_2 = np.transpose(features_sub1_eeg1_c0_reshape).reshape(-1)
    features_sub1_eeg2_c0_reshape_2 = np.transpose(features_sub1_eeg2_c0_reshape).reshape(-1)
    print('features_sub1_eeg1/eeg2_c0_reshape_2', features_sub1_eeg2_c0_reshape_2.shape)

    [num, f, t] = features_sub1_eeg1_c1.shape
    print('features_sub1_eeg1/eeg2_c1.shape', features_sub1_eeg1_c1.shape)
    features_sub1_eeg1_c1_reshape = features_sub1_eeg1_c1.reshape(f, t * num)
    features_sub1_eeg2_c1_reshape = features_sub1_eeg2_c1.reshape(f, t * num)
    print('features_sub1_eeg1/eeg2_c1_reshape', features_sub1_eeg1_c1_reshape.shape)
    features_sub1_eeg1_c1_reshape_2 = np.transpose(features_sub1_eeg1_c1_reshape).reshape(-1)
    features_sub1_eeg2_c1_reshape_2 = np.transpose(features_sub1_eeg2_c1_reshape).reshape(-1)
    print('features_sub1_eeg1/eeg2_c1_reshape_2', features_sub1_eeg2_c1_reshape_2.shape)

    [num, f, t] = features_sub1_eeg1_c2.shape
    print('features_sub1_eeg1/eeg2_c2.shape', features_sub1_eeg1_c2.shape)
    features_sub1_eeg1_c2_reshape = features_sub1_eeg1_c2.reshape(f, t * num)
    features_sub1_eeg2_c2_reshape = features_sub1_eeg2_c2.reshape(f, t * num)
    print('features_sub1_eeg1/eeg2_c2_reshape', features_sub1_eeg1_c2_reshape.shape)
    features_sub1_eeg1_c2_reshape_2 = np.transpose(features_sub1_eeg1_c2_reshape).reshape(-1)
    features_sub1_eeg2_c2_reshape_2 = np.transpose(features_sub1_eeg2_c2_reshape).reshape(-1)
    print('features_sub1_eeg1/eeg2_c2_reshape_2', features_sub1_eeg2_c2_reshape_2.shape)

    [num, f, t] = features_sub2_eeg1_c0.shape
    print('features_sub2_eeg1/eeg2_c0.shape', features_sub2_eeg1_c0.shape)
    features_sub2_eeg1_c0_reshape = features_sub2_eeg1_c0.reshape(f, t * num)
    features_sub2_eeg2_c0_reshape = features_sub2_eeg2_c0.reshape(f, t * num)
    print('features_sub2_eeg1/eeg2_c0_reshape', features_sub2_eeg1_c0_reshape.shape)
    features_sub2_eeg1_c0_reshape_2 = np.transpose(features_sub2_eeg1_c0_reshape).reshape(-1)
    features_sub2_eeg2_c0_reshape_2 = np.transpose(features_sub2_eeg2_c0_reshape).reshape(-1)
    print('features_sub2_eeg1/eeg2_c0_reshape_2', features_sub2_eeg2_c0_reshape_2.shape)

    [num, f, t] = features_sub2_eeg1_c1.shape
    print('features_sub2_eeg1/eeg2_c1.shape', features_sub2_eeg1_c1.shape)
    features_sub2_eeg1_c1_reshape = features_sub2_eeg1_c1.reshape(f, t * num)
    features_sub2_eeg2_c1_reshape = features_sub2_eeg2_c1.reshape(f, t * num)
    print('features_sub2_eeg1/eeg2_c1_reshape', features_sub2_eeg1_c1_reshape.shape)
    features_sub2_eeg1_c1_reshape_2 = np.transpose(features_sub2_eeg1_c1_reshape).reshape(-1)
    features_sub2_eeg2_c1_reshape_2 = np.transpose(features_sub2_eeg2_c1_reshape).reshape(-1)
    print('features_sub2_eeg1/eeg2_c1_reshape_2', features_sub2_eeg2_c1_reshape_2.shape)

    [num, f, t] = features_sub2_eeg1_c2.shape
    print('features_sub2_eeg1/eeg2_c2.shape', features_sub2_eeg1_c2.shape)
    features_sub2_eeg1_c2_reshape = features_sub2_eeg1_c2.reshape(f, t * num)
    features_sub2_eeg2_c2_reshape = features_sub2_eeg2_c2.reshape(f, t * num)
    print('features_sub2_eeg1/eeg2_c2_reshape', features_sub2_eeg1_c2_reshape.shape)
    features_sub2_eeg1_c2_reshape_2 = np.transpose(features_sub2_eeg1_c2_reshape).reshape(-1)
    features_sub2_eeg2_c2_reshape_2 = np.transpose(features_sub2_eeg2_c2_reshape).reshape(-1)
    print('features_sub2_eeg1/eeg2_c2_reshape_2', features_sub2_eeg2_c2_reshape_2.shape)

    # concatenation all with the same eeg channel and class
    # for convenience of seaborn.lineplot()
    features_eeg1_c0 = np.concatenate((features_sub0_eeg1_c0_reshape_2,features_sub1_eeg1_c0_reshape_2,features_sub2_eeg1_c0_reshape_2))
    features_eeg2_c0 = np.concatenate((features_sub0_eeg2_c0_reshape_2,features_sub1_eeg2_c0_reshape_2,features_sub2_eeg2_c0_reshape_2))
    features_eeg1_c1 = np.concatenate((features_sub0_eeg1_c1_reshape_2,features_sub1_eeg1_c1_reshape_2,features_sub2_eeg1_c1_reshape_2))
    features_eeg2_c1 = np.concatenate((features_sub0_eeg2_c1_reshape_2,features_sub1_eeg2_c1_reshape_2,features_sub2_eeg2_c1_reshape_2))
    features_eeg1_c2 = np.concatenate((features_sub0_eeg1_c2_reshape_2,features_sub1_eeg1_c2_reshape_2,features_sub2_eeg1_c2_reshape_2))
    features_eeg2_c2 = np.concatenate((features_sub0_eeg2_c2_reshape_2,features_sub1_eeg2_c2_reshape_2,features_sub2_eeg2_c2_reshape_2))
    print('features_eeg1_c0.shape',features_eeg1_c0.shape)
    # number for each subject is n_subi_cj * 48 * 32
    sub_label_c0 = np.concatenate((np.full((n_sub0_c0*48*32),'sub0'),np.full((n_sub1_c0*48*32),'sub1'),np.full((n_sub2_c0*48*32),'sub2')))
    sub_label_c1 = np.concatenate((np.full((n_sub0_c1*48*32),'sub0'),np.full((n_sub1_c1*48*32),'sub1'),np.full((n_sub2_c1*48*32),'sub2')))
    sub_label_c2 = np.concatenate((np.full((n_sub0_c2*48*32),'sub0'),np.full((n_sub1_c2*48*32),'sub1'),np.full((n_sub2_c2*48*32),'sub2')))
    print('sub_label_c0.shape',sub_label_c0.shape)
    # number for each freq is
    freq = np.arange(start=0.5,stop=24.5,step=0.5)
    freq_c0 = np.repeat(freq, (n_sub0_c0+n_sub1_c0+n_sub2_c0)*32)
    freq_c1 = np.repeat(freq, (n_sub0_c1+n_sub1_c1+n_sub2_c1)*32)
    freq_c2 = np.repeat(freq, (n_sub0_c2+n_sub1_c2+n_sub2_c2)*32)
    print('freq_c0.shape',freq_c0.shape)
    
    # write PSD's to DataFrame
    dict_PSD_eeg1_c0_sub0 = {'freq':np.repeat(freq, n_sub0_c0*32), 
                             'psd':features_sub0_eeg1_c0_reshape_2,
                             'subj':np.full((n_sub0_c0*48*32),'sub0')} 
    dict_PSD_eeg1_c0_sub1 = {'freq':np.repeat(freq, n_sub1_c0*32), 
                             'psd':features_sub1_eeg1_c0_reshape_2,
                             'subj':np.full((n_sub1_c0*48*32),'sub1')} 
    dict_PSD_eeg1_c0_sub2 = {'freq':np.repeat(freq, n_sub2_c0*32), 
                             'psd':features_sub2_eeg1_c0_reshape_2,
                             'subj':np.full((n_sub2_c0*48*32),'sub1')} 
    df_PSD_eeg1_c0_sub0 = pd.DataFrame.from_dict(dict_PSD_eeg1_c0_sub0)
    df_PSD_eeg1_c0_sub1 = pd.DataFrame.from_dict(dict_PSD_eeg1_c0_sub1)
    df_PSD_eeg1_c0_sub2 = pd.DataFrame.from_dict(dict_PSD_eeg1_c0_sub2)

    dict_PSD_eeg2_c0_sub0 = {'freq':np.repeat(freq, n_sub0_c0*32), 
                             'psd':features_sub0_eeg2_c0_reshape_2,
                             'subj':np.full((n_sub0_c0*48*32),'sub0')} 
    dict_PSD_eeg2_c0_sub1 = {'freq':np.repeat(freq, n_sub1_c0*32), 
                             'psd':features_sub1_eeg2_c0_reshape_2,
                             'subj':np.full((n_sub1_c0*48*32),'sub1')} 
    dict_PSD_eeg2_c0_sub2 = {'freq':np.repeat(freq, n_sub2_c0*32), 
                             'psd':features_sub2_eeg2_c0_reshape_2,
                             'subj':np.full((n_sub2_c0*48*32),'sub1')} 
    df_PSD_eeg2_c0_sub0 = pd.DataFrame.from_dict(dict_PSD_eeg2_c0_sub0)
    df_PSD_eeg2_c0_sub1 = pd.DataFrame.from_dict(dict_PSD_eeg2_c0_sub1)
    df_PSD_eeg2_c0_sub2 = pd.DataFrame.from_dict(dict_PSD_eeg2_c0_sub2)

    dict_PSD_eeg1_c1_sub0 = {'freq':np.repeat(freq, n_sub0_c1*32), 
                             'psd':features_sub0_eeg1_c1_reshape_2,
                             'subj':np.full((n_sub0_c1*48*32),'sub0')} 
    dict_PSD_eeg1_c1_sub1 = {'freq':np.repeat(freq, n_sub1_c1*32), 
                             'psd':features_sub1_eeg1_c1_reshape_2,
                             'subj':np.full((n_sub1_c1*48*32),'sub1')} 
    dict_PSD_eeg1_c1_sub2 = {'freq':np.repeat(freq, n_sub2_c1*32), 
                             'psd':features_sub2_eeg1_c1_reshape_2,
                             'subj':np.full((n_sub2_c1*48*32),'sub1')} 
    df_PSD_eeg1_c1_sub0 = pd.DataFrame.from_dict(dict_PSD_eeg1_c1_sub0)
    df_PSD_eeg1_c1_sub1 = pd.DataFrame.from_dict(dict_PSD_eeg1_c1_sub1)
    df_PSD_eeg1_c1_sub2 = pd.DataFrame.from_dict(dict_PSD_eeg1_c1_sub2)

    dict_PSD_eeg2_c1_sub0 = {'freq':np.repeat(freq, n_sub0_c1*32), 
                             'psd':features_sub0_eeg2_c1_reshape_2,
                             'subj':np.full((n_sub0_c1*48*32),'sub0')} 
    dict_PSD_eeg2_c1_sub1 = {'freq':np.repeat(freq, n_sub1_c1*32), 
                             'psd':features_sub1_eeg2_c1_reshape_2,
                             'subj':np.full((n_sub1_c1*48*32),'sub1')} 
    dict_PSD_eeg2_c1_sub2 = {'freq':np.repeat(freq, n_sub2_c1*32), 
                             'psd':features_sub2_eeg2_c1_reshape_2,
                             'subj':np.full((n_sub2_c1*48*32),'sub1')} 
    df_PSD_eeg2_c1_sub0 = pd.DataFrame.from_dict(dict_PSD_eeg2_c1_sub0)
    df_PSD_eeg2_c1_sub1 = pd.DataFrame.from_dict(dict_PSD_eeg2_c1_sub1)
    df_PSD_eeg2_c1_sub2 = pd.DataFrame.from_dict(dict_PSD_eeg2_c1_sub2)

    dict_PSD_eeg1_c2_sub0 = {'freq':np.repeat(freq, n_sub0_c2*32), 
                             'psd':features_sub0_eeg1_c2_reshape_2,
                             'subj':np.full((n_sub0_c2*48*32),'sub0')} 
    dict_PSD_eeg1_c2_sub1 = {'freq':np.repeat(freq, n_sub1_c2*32), 
                             'psd':features_sub1_eeg1_c2_reshape_2,
                             'subj':np.full((n_sub1_c2*48*32),'sub1')} 
    dict_PSD_eeg1_c2_sub2 = {'freq':np.repeat(freq, n_sub2_c2*32), 
                             'psd':features_sub2_eeg1_c2_reshape_2,
                             'subj':np.full((n_sub2_c2*48*32),'sub1')} 
    df_PSD_eeg1_c2_sub0 = pd.DataFrame.from_dict(dict_PSD_eeg1_c2_sub0)
    df_PSD_eeg1_c2_sub1 = pd.DataFrame.from_dict(dict_PSD_eeg1_c2_sub1)
    df_PSD_eeg1_c2_sub2 = pd.DataFrame.from_dict(dict_PSD_eeg1_c2_sub2)

    dict_PSD_eeg2_c2_sub0 = {'freq':np.repeat(freq, n_sub0_c2*32), 
                             'psd':features_sub0_eeg2_c2_reshape_2,
                             'subj':np.full((n_sub0_c2*48*32),'sub0')} 
    dict_PSD_eeg2_c2_sub1 = {'freq':np.repeat(freq, n_sub1_c2*32), 
                             'psd':features_sub1_eeg2_c2_reshape_2,
                             'subj':np.full((n_sub1_c2*48*32),'sub1')} 
    dict_PSD_eeg2_c2_sub2 = {'freq':np.repeat(freq, n_sub2_c2*32), 
                             'psd':features_sub2_eeg2_c2_reshape_2,
                             'subj':np.full((n_sub2_c2*48*32),'sub1')} 
    df_PSD_eeg2_c2_sub0 = pd.DataFrame.from_dict(dict_PSD_eeg2_c2_sub0)
    df_PSD_eeg2_c2_sub1 = pd.DataFrame.from_dict(dict_PSD_eeg2_c2_sub1)
    df_PSD_eeg2_c2_sub2 = pd.DataFrame.from_dict(dict_PSD_eeg2_c2_sub2)

    # save df
    if persub:
        print('mode: persub')
        df_PSD_eeg1_c0_sub0.to_pickle('./data/PSD_eeg1_c0_sub0.pkl')
        df_PSD_eeg1_c1_sub0.to_pickle('./data/PSD_eeg1_c1_sub0.pkl')
        df_PSD_eeg1_c2_sub0.to_pickle('./data/PSD_eeg1_c2_sub0.pkl')

        df_PSD_eeg2_c0_sub0.to_pickle('./data/PSD_eeg2_c0_sub0.pkl')
        df_PSD_eeg2_c1_sub0.to_pickle('./data/PSD_eeg2_c1_sub0.pkl')
        df_PSD_eeg2_c2_sub0.to_pickle('./data/PSD_eeg2_c2_sub0.pkl')

        df_PSD_eeg1_c0_sub1.to_pickle('./data/PSD_eeg1_c0_sub1.pkl')
        df_PSD_eeg1_c1_sub1.to_pickle('./data/PSD_eeg1_c1_sub1.pkl')
        df_PSD_eeg1_c2_sub1.to_pickle('./data/PSD_eeg1_c2_sub1.pkl')

        df_PSD_eeg2_c0_sub1.to_pickle('./data/PSD_eeg2_c0_sub1.pkl')
        df_PSD_eeg2_c1_sub1.to_pickle('./data/PSD_eeg2_c1_sub1.pkl')
        df_PSD_eeg2_c2_sub1.to_pickle('./data/PSD_eeg2_c2_sub1.pkl')

        df_PSD_eeg1_c0_sub2.to_pickle('./data/PSD_eeg1_c0_sub2.pkl')
        df_PSD_eeg1_c1_sub2.to_pickle('./data/PSD_eeg1_c1_sub2.pkl')
        df_PSD_eeg1_c2_sub2.to_pickle('./data/PSD_eeg1_c2_sub2.pkl')

        df_PSD_eeg2_c0_sub2.to_pickle('./data/PSD_eeg2_c0_sub2.pkl')
        df_PSD_eeg2_c1_sub2.to_pickle('./data/PSD_eeg2_c1_sub2.pkl')
        df_PSD_eeg2_c2_sub2.to_pickle('./data/PSD_eeg2_c2_sub2.pkl')
    elif persub==False:
        print('mode: no persub')
        df_PSD_eeg1_c0_sub0.to_pickle('./data/PSD_eeg1_c0_sub0_no_pb.pkl')
        df_PSD_eeg1_c1_sub0.to_pickle('./data/PSD_eeg1_c1_sub0_no_pb.pkl')
        df_PSD_eeg1_c2_sub0.to_pickle('./data/PSD_eeg1_c2_sub0_no_pb.pkl')

        df_PSD_eeg2_c0_sub0.to_pickle('./data/PSD_eeg2_c0_sub0_no_pb.pkl')
        df_PSD_eeg2_c1_sub0.to_pickle('./data/PSD_eeg2_c1_sub0_no_pb.pkl')
        df_PSD_eeg2_c2_sub0.to_pickle('./data/PSD_eeg2_c2_sub0_no_pb.pkl')

        df_PSD_eeg1_c0_sub1.to_pickle('./data/PSD_eeg1_c0_sub1_no_pb.pkl')
        df_PSD_eeg1_c1_sub1.to_pickle('./data/PSD_eeg1_c1_sub1_no_pb.pkl')
        df_PSD_eeg1_c2_sub1.to_pickle('./data/PSD_eeg1_c2_sub1_no_pb.pkl')

        df_PSD_eeg2_c0_sub1.to_pickle('./data/PSD_eeg2_c0_sub1_no_pb.pkl')
        df_PSD_eeg2_c1_sub1.to_pickle('./data/PSD_eeg2_c1_sub1_no_pb.pkl')
        df_PSD_eeg2_c2_sub1.to_pickle('./data/PSD_eeg2_c2_sub1_no_pb.pkl')

        df_PSD_eeg1_c0_sub2.to_pickle('./data/PSD_eeg1_c0_sub2_no_pb.pkl')
        df_PSD_eeg1_c1_sub2.to_pickle('./data/PSD_eeg1_c1_sub2_no_pb.pkl')
        df_PSD_eeg1_c2_sub2.to_pickle('./data/PSD_eeg1_c2_sub2_no_pb.pkl')

        df_PSD_eeg2_c0_sub2.to_pickle('./data/PSD_eeg2_c0_sub2_no_pb.pkl')
        df_PSD_eeg2_c1_sub2.to_pickle('./data/PSD_eeg2_c1_sub2_no_pb.pkl')
        df_PSD_eeg2_c2_sub2.to_pickle('./data/PSD_eeg2_c2_sub2_no_pb.pkl')

def print_visualization(persub):
    # df of PSD
    if persub:
        df_PSD_eeg1_c0_sub0 = pd.read_pickle('./data/PSD_eeg1_c0_sub0.pkl')
        df_PSD_eeg1_c1_sub0 = pd.read_pickle('./data/PSD_eeg1_c1_sub0.pkl')
        df_PSD_eeg1_c2_sub0 = pd.read_pickle('./data/PSD_eeg1_c2_sub0.pkl')

        df_PSD_eeg2_c0_sub0 = pd.read_pickle('./data/PSD_eeg2_c0_sub0.pkl')
        df_PSD_eeg2_c1_sub0 = pd.read_pickle('./data/PSD_eeg2_c1_sub0.pkl')
        df_PSD_eeg2_c2_sub0 = pd.read_pickle('./data/PSD_eeg2_c2_sub0.pkl')

        df_PSD_eeg1_c0_sub1 = pd.read_pickle('./data/PSD_eeg1_c0_sub1.pkl')
        df_PSD_eeg1_c1_sub1 = pd.read_pickle('./data/PSD_eeg1_c1_sub1.pkl')
        df_PSD_eeg1_c2_sub1 = pd.read_pickle('./data/PSD_eeg1_c2_sub1.pkl')

        df_PSD_eeg2_c0_sub1 = pd.read_pickle('./data/PSD_eeg2_c0_sub1.pkl')
        df_PSD_eeg2_c1_sub1 = pd.read_pickle('./data/PSD_eeg2_c1_sub1.pkl')
        df_PSD_eeg2_c2_sub1 = pd.read_pickle('./data/PSD_eeg2_c2_sub1.pkl')

        df_PSD_eeg1_c0_sub2 = pd.read_pickle('./data/PSD_eeg1_c0_sub2.pkl')
        df_PSD_eeg1_c1_sub2 = pd.read_pickle('./data/PSD_eeg1_c1_sub2.pkl')
        df_PSD_eeg1_c2_sub2 = pd.read_pickle('./data/PSD_eeg1_c2_sub2.pkl')

        df_PSD_eeg2_c0_sub2 = pd.read_pickle('./data/PSD_eeg2_c0_sub2.pkl')
        df_PSD_eeg2_c1_sub2 = pd.read_pickle('./data/PSD_eeg2_c1_sub2.pkl')
        df_PSD_eeg2_c2_sub2 = pd.read_pickle('./data/PSD_eeg2_c2_sub2.pkl')

    elif persub==False:
        df_PSD_eeg1_c0_sub0 = pd.read_pickle('./data/PSD_eeg1_c0_sub0_no_pb.pkl')
        df_PSD_eeg1_c1_sub0 = pd.read_pickle('./data/PSD_eeg1_c1_sub0_no_pb.pkl')
        df_PSD_eeg1_c2_sub0 = pd.read_pickle('./data/PSD_eeg1_c2_sub0_no_pb.pkl')

        df_PSD_eeg2_c0_sub0 = pd.read_pickle('./data/PSD_eeg2_c0_sub0_no_pb.pkl')
        df_PSD_eeg2_c1_sub0 = pd.read_pickle('./data/PSD_eeg2_c1_sub0_no_pb.pkl')
        df_PSD_eeg2_c2_sub0 = pd.read_pickle('./data/PSD_eeg2_c2_sub0_no_pb.pkl')

        df_PSD_eeg1_c0_sub1 = pd.read_pickle('./data/PSD_eeg1_c0_sub1_no_pb.pkl')
        df_PSD_eeg1_c1_sub1 = pd.read_pickle('./data/PSD_eeg1_c1_sub1_no_pb.pkl')
        df_PSD_eeg1_c2_sub1 = pd.read_pickle('./data/PSD_eeg1_c2_sub1_no_pb.pkl')

        df_PSD_eeg2_c0_sub1 = pd.read_pickle('./data/PSD_eeg2_c0_sub1_no_pb.pkl')
        df_PSD_eeg2_c1_sub1 = pd.read_pickle('./data/PSD_eeg2_c1_sub1_no_pb.pkl')
        df_PSD_eeg2_c2_sub1 = pd.read_pickle('./data/PSD_eeg2_c2_sub1_no_pb.pkl')

        df_PSD_eeg1_c0_sub2 = pd.read_pickle('./data/PSD_eeg1_c0_sub2_no_pb.pkl')
        df_PSD_eeg1_c1_sub2 = pd.read_pickle('./data/PSD_eeg1_c1_sub2_no_pb.pkl')
        df_PSD_eeg1_c2_sub2 = pd.read_pickle('./data/PSD_eeg1_c2_sub2_no_pb.pkl')

        df_PSD_eeg2_c0_sub2 = pd.read_pickle('./data/PSD_eeg2_c0_sub2_no_pb.pkl')
        df_PSD_eeg2_c1_sub2 = pd.read_pickle('./data/PSD_eeg2_c1_sub2_no_pb.pkl')
        df_PSD_eeg2_c2_sub2 = pd.read_pickle('./data/PSD_eeg2_c2_sub2_no_pb.pkl')
    # plot
    # TO DO: plot every mean vector of features
    fig1 = plt.figure()
    sns.lineplot(data=df_PSD_eeg1_c0_sub0, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg1_c0_sub1, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg1_c0_sub2, x='freq', y='psd')
    if persub:
        fig1.savefig('./PSD_fig/PSD_eeg1_c0.png')
    elif persub==False:
        fig1.savefig('./PSD_fig_no_pb/PSD_eeg1_c0_no_pb.png')
    fig2 = plt.figure()
    sns.lineplot(data=df_PSD_eeg2_c0_sub0, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg2_c0_sub1, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg2_c0_sub2, x='freq', y='psd')
    if persub:
        fig2.savefig('./PSD_fig/PSD_eeg2_c0.png')
    elif persub==False:
        fig2.savefig('./PSD_fig_no_pb/PSD_eeg2_c0_no_pb.png')

    fig3 = plt.figure()
    sns.lineplot(data=df_PSD_eeg1_c1_sub0, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg1_c1_sub1, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg1_c1_sub2, x='freq', y='psd')
    if persub:
        fig3.savefig('./PSD_fig/PSD_eeg1_c1.png')
    elif persub==False:
        fig3.savefig('./PSD_fig_no_pb/PSD_eeg1_c1_no_pb.png')
    fig4 = plt.figure()
    sns.lineplot(data=df_PSD_eeg2_c1_sub0, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg2_c1_sub1, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg2_c1_sub2, x='freq', y='psd')
    if persub:
        fig4.savefig('./PSD_fig/PSD_eeg2_c1.png')
    elif persub==False:
        fig4.savefig('./PSD_fig_no_pb/PSD_eeg2_c1_no_pb.png')

    fig5 = plt.figure()
    sns.lineplot(data=df_PSD_eeg1_c2_sub0, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg1_c2_sub1, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg1_c2_sub2, x='freq', y='psd')
    if persub:
        fig5.savefig('./PSD_fig/PSD_eeg1_c2.png')
    elif persub==False:
        fig5.savefig('./PSD_fig_no_pb/PSD_eeg1_c2_no_pb.png')
    fig6 = plt.figure()
    sns.lineplot(data=df_PSD_eeg2_c2_sub0, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg2_c2_sub1, x='freq', y='psd')
    sns.lineplot(data=df_PSD_eeg2_c2_sub2, x='freq', y='psd')
    if persub:
        fig6.savefig('./PSD_fig/PSD_eeg2_c2.png')
    elif persub==False:
        fig6.savefig('./PSD_fig_no_pb/PSD_eeg2_c2_no_pb.png')


def main():
    ###
    # before running this code, please first run 'csv2npy.py' and 'prepare_fft_feature.py'
    # generate npy files in './data/'
    ###
    print()
    print('***************By Killer Queen***************')
    # w/ per subject normalization
    #generate_per_sub_features()  # (Can be commented out after running once)
    # w/o per subject normalization
    #generate_without_per_sub_features()  # (Can be commented out after running once)
    # PSD
    gen_PSD_data(persub=False)   # (Can be commented out after running once)
    # plot PSD
    print_visualization(persub=False)

main()
