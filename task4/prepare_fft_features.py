import numpy as np
import os
from scipy import signal


def extract_and_save_fft(signals, name, mode, cut_len):
    print('Processing signal {} {}...'.format(mode, name))
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if mode == 'train':
        sub_num = 3
    else:
        sub_num = 2
    fft_features = []
    for i in range(sub_num):
        sub_signals = signals[i*21600:(i+1)*21600]
        sig_pad_1 = sub_signals[0, :240].copy()
        sig_pad_2 = sub_signals[-1, -240:].copy()
        sig_concat = sub_signals.flatten()
        sig_concat = np.concatenate((sig_pad_1, sig_concat, sig_pad_2))
        f, t, Sxx = signal.spectrogram(sig_concat, fs=128, window='hamming', noverlap=256 - 16, nperseg=256)
        Sxx = Sxx[1:cut_len + 1, :]
        for i in range(21600):
            fft_features.append(Sxx[:, i*32:(i+1)*32])
    fft_features = np.array(fft_features)
    np.save('./data/{}_{}_fft_features_new.npy'.format(mode, name), fft_features)
    return


def main():
    print()
    print('***************By Killer Queen***************')

    sig = np.load('./data/train_eeg1.npy')
    extract_and_save_fft(sig, 'eeg1', 'train', 48)
    sig = np.load('./data/train_eeg2.npy')
    extract_and_save_fft(sig, 'eeg2', 'train', 48)
    sig = np.load('./data/train_emg.npy')
    extract_and_save_fft(sig, 'emg', 'train', 60)
    sig = np.load('./data/test_eeg1.npy')
    extract_and_save_fft(sig, 'eeg1', 'test', 48)
    sig = np.load('./data/test_eeg2.npy')
    extract_and_save_fft(sig, 'eeg2', 'test', 48)
    sig = np.load('./data/test_emg.npy')
    extract_and_save_fft(sig, 'emg', 'test', 60)


main()
