import numpy as np
from tqdm import tqdm
import os
import math


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
        # print('eeg', eeg1)
        # print(eeg1.shape)
        # eeg1_reshape = eeg1.reshape(48, 32*64800)
        # print('eeg_reshape', eeg1_reshape)
        # print(eeg1_reshape.shape)
        eeg2 = np.log(eeg2)
        # eeg2_reshape = eeg2.reshape(48, 32*64800)
        emg = np.log(emg)
        print(eeg1.shape)
        print(eeg2.shape)
        print(emg.shape)
        for eeg1_sig, eeg2_sig, emg_sig in tqdm(zip(eeg1, eeg2, emg)):
            # print('eeg1_sig', eeg1_sig.shape)
            # print(eeg1_sig)
            # print('np.mean', np.mean(eeg1_sig, axis=1, keepdims=True))
            # print('np.mean', np.mean(eeg1_sig, axis=1, keepdims=True).shape)
            # print('np.mean', np.mean(eeg1_sig, axis=0, keepdims=True).shape)
            # print('np.std', np.std(eeg1_sig, axis=1, keepdims=True))
            eeg1_sig = (eeg1_sig - np.mean(eeg1_sig, axis=1, keepdims=True)) / np.std(eeg1_sig, axis=1, keepdims=True)
            # print('eeg1_sig', eeg1_sig.shape) # eeg1_sig (48, 32)
            eeg2_sig = (eeg2_sig - np.mean(eeg2_sig, axis=1, keepdims=True)) / np.std(eeg2_sig, axis=1, keepdims=True)
            # print('eeg2_sig', eeg2_sig.shape) # eeg2_sig (48, 32)
            emg_sig = np.sum(emg_sig, axis=0, keepdims=True)
            # print('emg_sig', emg_sig.shape) # emg_sig (1, 32)
            emg_sig = emg_sig.repeat(eeg1_sig.shape[0], axis=0)
            # print('emg_sig_after', emg_sig.shape) # emg_sig_after (48, 32)
            eeg1_sig = eeg1_sig.reshape((eeg1_sig.shape[0], eeg1_sig.shape[1], 1))
            eeg2_sig = eeg2_sig.reshape((eeg1_sig.shape[0], eeg1_sig.shape[1], 1))
            emg_sig = emg_sig.reshape((eeg1_sig.shape[0], eeg1_sig.shape[1], 1))
            # eeg1_sig = np.log(eeg1_sig)
            # eeg2_sig = np.log(eeg2_sig)
            # emg_sig = np.log(emg_sig)
            # print(eeg1_sig.shape)
            # print(eeg2_sig.shape)
            # print(emg_sig.shape)
            features.append(np.concatenate((eeg1_sig, eeg2_sig, emg_sig), axis=2))
        features = np.array(features)
        np.save('./data/{}_x.npy'.format(mode), np.array(features))


def per_subject_normalization_train():
    print('Per-subject Normalization train...')
    # modes = ['train', 'test']
    mode = 'train'
    # for mode in modes:
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

    print(eeg1.shape)
    print(eeg2.shape)
    print(emg.shape)

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
        eeg1_sub0_sig = eeg1_sub0_sig.reshape((eeg1_sub0_sig.shape[0], eeg1_sub0_sig.shape[1], 1))
        eeg2_sub0_sig = eeg2_sub0_sig.reshape((eeg1_sub0_sig.shape[0], eeg1_sub0_sig.shape[1], 1))
        emg_sub0_sig = emg_sub0_sig.reshape((eeg1_sub0_sig.shape[0], eeg1_sub0_sig.shape[1], 1))
        # print(eeg1_sub0_sig.shape)
        # print(eeg2_sub0_sig.shape)
        # print(emg_sub0_sig.shape)
        features.append(np.concatenate((eeg1_sub0_sig, eeg2_sub0_sig, emg_sub0_sig), axis=2))

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
        eeg1_sub1_sig = eeg1_sub1_sig.reshape((eeg1_sub1_sig.shape[0], eeg1_sub1_sig.shape[1], 1))
        eeg2_sub1_sig = eeg2_sub1_sig.reshape((eeg1_sub1_sig.shape[0], eeg1_sub1_sig.shape[1], 1))
        emg_sub1_sig = emg_sub1_sig.reshape((eeg1_sub1_sig.shape[0], eeg1_sub1_sig.shape[1], 1))
        # print(eeg1_sub1_sig.shape)
        # print(eeg2_sub1_sig.shape)
        # print(emg_sub1_sig.shape)
        features.append(np.concatenate((eeg1_sub1_sig, eeg2_sub1_sig, emg_sub1_sig), axis=2))

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
        eeg1_sub2_sig = eeg1_sub2_sig.reshape((eeg1_sub2_sig.shape[0], eeg1_sub2_sig.shape[1], 1))
        eeg2_sub2_sig = eeg2_sub2_sig.reshape((eeg1_sub2_sig.shape[0], eeg1_sub2_sig.shape[1], 1))
        emg_sub2_sig = emg_sub2_sig.reshape((eeg1_sub2_sig.shape[0], eeg1_sub2_sig.shape[1], 1))
        # print(eeg1_sub2_sig.shape)
        # print(eeg2_sub2_sig.shape)
        # print(emg_sub2_sig.shape)
        features.append(np.concatenate((eeg1_sub2_sig, eeg2_sub2_sig, emg_sub2_sig), axis=2))

    features = np.array(features)
    np.save('./data/{}_x.npy'.format(mode), np.array(features))


def per_subject_normalization_test():
    print('Per-subject Normalization test...')
    # modes = ['train', 'test']
    mode = 'test'
    # for mode in modes:
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

    print(eeg1.shape)
    print(eeg2.shape)
    print(emg.shape)

    eeg1_sub0 = eeg1[:21600, :, :]
    eeg1_sub1 = eeg1[21600:2 * 21600, :, :]

    print('eeg1_sub0', eeg1_sub0.shape)
    print('eeg1_sub1', eeg1_sub1.shape)

    eeg2_sub0 = eeg2[:21600, :, :]
    eeg2_sub1 = eeg2[21600:2 * 21600, :, :]

    emg_sub0 = emg[:21600, :, :]
    emg_sub1 = emg[21600:2 * 21600, :, :]

    eeg1_sub0_reshape = eeg1_sub0.reshape(48, 32 * 21600)
    eeg1_sub1_reshape = eeg1_sub1.reshape(48, 32 * 21600)

    eeg1_sub0_reshape_mean = np.mean(eeg1_sub0_reshape, axis=1, keepdims=True)
    eeg1_sub0_reshape_std = np.std(eeg1_sub0_reshape, axis=1, keepdims=True)
    eeg1_sub1_reshape_mean = np.mean(eeg1_sub1_reshape, axis=1, keepdims=True)
    eeg1_sub1_reshape_std = np.std(eeg1_sub1_reshape, axis=1, keepdims=True)

    eeg2_sub0_reshape = eeg2_sub0.reshape(48, 32 * 21600)
    eeg2_sub1_reshape = eeg2_sub1.reshape(48, 32 * 21600)

    eeg2_sub0_reshape_mean = np.mean(eeg2_sub0_reshape, axis=1, keepdims=True)
    eeg2_sub0_reshape_std = np.std(eeg2_sub0_reshape, axis=1, keepdims=True)
    eeg2_sub1_reshape_mean = np.mean(eeg2_sub1_reshape, axis=1, keepdims=True)
    eeg2_sub1_reshape_std = np.std(eeg2_sub1_reshape, axis=1, keepdims=True)

    emg_sub0_reshape = emg_sub0.reshape(60, 32 * 21600)
    emg_sub1_reshape = emg_sub1.reshape(60, 32 * 21600)

    for eeg1_sub0_sig, eeg2_sub0_sig, emg_sub0_sig in tqdm(zip(eeg1_sub0, eeg2_sub0, emg_sub0)):
        # print('eeg1_sub0_sig', eeg1_sub0_sig.shape)
        eeg1_sub0_sig = (eeg1_sub0_sig - eeg1_sub0_reshape_mean) / eeg1_sub0_reshape_std
        # print('eeg1_sub0_sig', eeg1_sub0_sig.shape)
        eeg2_sub0_sig = (eeg2_sub0_sig - eeg2_sub0_reshape_mean) / eeg2_sub0_reshape_std
        # print('eeg2_sub0_sig', eeg2_sub0_sig.shape)

        emg_sub0_sig = np.sum(emg_sub0_sig, axis=0, keepdims=True)
        # print('emg_sig', emg_sub0_sig.shape)
        emg_sub0_sig = emg_sub0_sig.repeat(eeg1_sub0_sig.shape[0], axis=0)
        # print('emg_sig_after', emg_sub0_sig.shape)
        eeg1_sub0_sig = eeg1_sub0_sig.reshape((eeg1_sub0_sig.shape[0], eeg1_sub0_sig.shape[1], 1))
        eeg2_sub0_sig = eeg2_sub0_sig.reshape((eeg1_sub0_sig.shape[0], eeg1_sub0_sig.shape[1], 1))
        emg_sub0_sig = emg_sub0_sig.reshape((eeg1_sub0_sig.shape[0], eeg1_sub0_sig.shape[1], 1))
        # print(eeg1_sub0_sig.shape)
        # print(eeg2_sub0_sig.shape)
        # print(emg_sub0_sig.shape)
        features.append(np.concatenate((eeg1_sub0_sig, eeg2_sub0_sig, emg_sub0_sig), axis=2))

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
        eeg1_sub1_sig = eeg1_sub1_sig.reshape((eeg1_sub1_sig.shape[0], eeg1_sub1_sig.shape[1], 1))
        eeg2_sub1_sig = eeg2_sub1_sig.reshape((eeg1_sub1_sig.shape[0], eeg1_sub1_sig.shape[1], 1))
        emg_sub1_sig = emg_sub1_sig.reshape((eeg1_sub1_sig.shape[0], eeg1_sub1_sig.shape[1], 1))
        # print(eeg1_sub1_sig.shape)
        # print(eeg2_sub1_sig.shape)
        # print(emg_sub1_sig.shape)
        features.append(np.concatenate((eeg1_sub1_sig, eeg2_sub1_sig, emg_sub1_sig), axis=2))

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
                idx += 1


def main():
    print()
    print('***************By Killer Queen***************')

    # normalization()
    per_subject_normalization_train()
    per_subject_normalization_test()
    generate_feature_with_adjacent_epochs()


main()
