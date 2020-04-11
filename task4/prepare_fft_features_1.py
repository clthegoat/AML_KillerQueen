import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def extract_and_save_fft_old(train_eeg1, train_eeg2, train_emg, mode):
    if not os.path.exists('./data'):
        os.mkdir('./data')
    sample_rate = 128
    window_size = 256
    step_size = 16
    slice_size = int((512 - window_size)/step_size)
    names = ['eeg1', 'eeg2', 'emg']
    for name, signals in zip(names, [train_eeg1, train_eeg2, train_emg]):
        print("Preparing {} {} fft features...".format(mode, name))
        fft_features = []
        if name in ['eeg1', 'eeg2']:
            cut_len = 48
        else:
            cut_len = 60
        for sig in tqdm(signals):
            sig_fft_feature = np.zeros((cut_len, slice_size+1))
            for slice_idx in range(slice_size+1):
                start_pos = slice_idx*step_size
                slice = sig[start_pos:start_pos+window_size]
                slice_fft = np.abs(np.fft.rfft(slice)/window_size)
                slice_fft = slice_fft[1:cut_len+1]
                sig_fft_feature[:, slice_idx] = np.reshape(slice_fft, (cut_len, 1))[:, 0]
            # print(sig_fft_feature.shape)
            fft_features.append(sig_fft_feature)
        fft_features = np.array(fft_features)
        np.save('./data/{}_{}_fft_features.npy'.format(mode, name), fft_features)
    return


def extract_and_save_fft(train_eeg1, train_eeg2, train_emg, mode):
    if not os.path.exists('./data'):
        os.mkdir('./data')
    sample_rate = 128
    window_size = 256
    step_size = 16
    slice_size = int((512 - window_size)/step_size)
    names = ['eeg1', 'eeg2', 'emg']
    for name, signals in zip(names, [train_eeg1, train_eeg2, train_emg]):
        print("Preparing {} {} fft features...".format(mode, name))
        fft_features = []
        if name in ['eeg1', 'eeg2']:
            cut_len = 48
        else:
            cut_len = 60
        for sig in tqdm(signals):
            plt.clf()
            Pxx, f, bins, im = plt.specgram(sig, NFFT=256, Fs=128, noverlap=256-16)
            fft_features.append(Pxx[1:cut_len+1, :])
        fft_features = np.array(fft_features)
        np.save('./data/{}_{}_fft_features_new.npy'.format(mode, name), fft_features)
    return


def main():
    print()
    print('***************By Killer Queen***************')

    train_eeg1 = np.load('./data/train_eeg1.npy')
    train_eeg2 = np.load('./data/train_eeg2.npy')
    train_emg = np.load('./data/train_emg.npy')

    test_eeg1 = np.load('./data/test_eeg1.npy')
    test_eeg2 = np.load('./data/test_eeg2.npy')
    test_emg = np.load('./data/test_emg.npy')

    extract_and_save_fft_old(train_eeg1, train_eeg2, train_emg, 'train')
    extract_and_save_fft_old(test_eeg1, test_eeg2, test_emg, 'test')

    return


main()
