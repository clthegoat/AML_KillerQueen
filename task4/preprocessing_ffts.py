import numpy as np
from tqdm import tqdm
import os


def normalization():
    print('Normalization...')
    modes = ['train', 'test']

    for mode in modes:
        features = []
        if mode == 'train':
            eeg1 = np.load('./data/train_eeg1_fft_features_new.npy')
            eeg2 = np.load('./data/train_eeg2_fft_features_new.npy')
            emg = np.load('./data/train_emg_fft_features_new.npy')
        else:
            eeg1 = np.load('./data/test_eeg1_fft_features_new.npy')
            eeg2 = np.load('./data/test_eeg2_fft_features_new.npy')
            emg = np.load('./data/test_emg_fft_features_new.npy')
        eeg1 = np.log(eeg1)
        eeg2 = np.log(eeg2)
        emg = np.log(emg)
        for eeg1_sig, eeg2_sig, emg_sig in tqdm(zip(eeg1, eeg2, emg)):
            eeg1_sig = (eeg1_sig - np.mean(eeg1_sig, axis=0, keepdims=True)) / np.std(eeg1_sig, axis=0, keepdims=True)
            eeg2_sig = (eeg2_sig - np.mean(eeg2_sig, axis=0, keepdims=True)) / np.std(eeg2_sig, axis=0, keepdims=True)
            emg_sig = np.sum(emg_sig, axis=0, keepdims=True)
            emg_sig = emg_sig.repeat(eeg1_sig.shape[0], axis=0)
            eeg1_sig = eeg1_sig.reshape((eeg1_sig.shape[0], eeg1_sig.shape[1], 1))
            eeg2_sig = eeg2_sig.reshape((eeg1_sig.shape[0], eeg1_sig.shape[1], 1))
            emg_sig = emg_sig.reshape((eeg1_sig.shape[0], eeg1_sig.shape[1], 1))
            features.append(np.concatenate((eeg1_sig, eeg2_sig, emg_sig), axis=2))
        features = np.array(features)
        np.save('./data/{}_x.npy'.format(mode), np.array(features))


def generate_feature_with_adjacent_epochs():
    print('concat adjacent epochs feature')
    modes = ['train', 'test']
    for mode in modes:
        base_dir = './data/features_{}'.format(mode)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        features = np.load('./data/{}_x.npy'.format(mode))
        idx = 0
        if mode == 'train':
            sub_num = 3
        else:
            sub_num = 2
        for i in range(sub_num):
            sub_feature = features[i * 21600:(i + 1) * 21600]
            sub_feature[:, :, :, 2] = (sub_feature[:, :, :, 2] - np.mean(sub_feature[:, :, :, 2])) / np.std(
                sub_feature[:, :, :, 2])
            for j in tqdm(range(len(sub_feature))):
                lf = np.concatenate((sub_feature[j - 2], sub_feature[j - 1], sub_feature[j],
                                     sub_feature[(j + 1) % len(sub_feature)], sub_feature[(j + 2) % len(sub_feature)]),
                                    axis=1)
                np.save(os.path.join(base_dir, '{}.npy'.format(idx)), np.array(lf))
                idx+=1


def main():
    print()
    print('***************By Killer Queen***************')

    normalization()
    generate_feature_with_adjacent_epochs()


main()
