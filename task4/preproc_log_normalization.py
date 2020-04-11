import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras import optimizers
from keras import initializers

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv3D, Flatten, MaxPooling2D

# load spectrogram data
train_eeg1_fft = np.load('./data/train_eeg1_fft_features.npy')  #(64800, 49, 17)
train_eeg2_fft = np.load('./data/train_eeg2_fft_features.npy')  #(64800, 49, 17)
train_emg_fft = np.load('./data/train_emg_fft_features.npy')  #(64800, 61, 17)

test_eeg1_fft = np.load('./data/test_eeg1_fft_features.npy')  #(43200, 49, 17)
test_eeg2_fft = np.load('./data/test_eeg2_fft_features.npy')  #(43200, 49, 17)
test_emg_fft = np.load('./data/test_emg_fft_features.npy')  #(43200, 61, 17)

# dimensions
time_dim = train_eeg1_fft.shape[2]
freq_eeg_dim = train_eeg1_fft.shape[1]
freq_emg_dim = train_emg_fft.shape[1]
sample_train_dim = train_eeg1_fft.shape[0]
sample_test_dim = test_eeg2_fft.shape[0]

# functions
def calc_mean(arr, arr_type):
	if arr_type == 'train':
		mean = np.sum(np.sum(arr, axis=2), axis=0) / (time_dim * sample_train_dim / 3)
	else:
		mean = np.sum(np.sum(arr, axis=2), axis=0) / (time_dim * sample_test_dim / 2)
	mean = np.expand_dims(mean, axis=2)
	mean = np.expand_dims(mean, axis=0)
	return mean

def calc_std(arr, arr_type, mean):
	arr = arr - mean
	if arr_type == 'train':
		std = np.sqrt(np.sum(np.sum(np.square(arr), axis=2), axis=0) / (time_dim * sample_train_dim / 3))
	else:
		std = np.sqrt(np.sum(np.sum(np.square(arr), axis=2), axis=0) / (time_dim * sample_test_dim / 2))
	std = np.expand_dims(std, axis=2)
	std = np.expand_dims(std, axis=0)
	return std

def normalize(arr, mean, std):
	arr = (arr - mean) / std
	#for i in range(arr.shape[0]):
	#	for j in range(arr.shape[2]):
	#		arr[i,:,j] = (arr[i,:,j] - mean) / std
	return arr

## eeg preprocessing
#------------------------------------------
# log
train_eeg1_fft = np.log(train_eeg1_fft)
train_eeg2_fft = np.log(train_eeg2_fft)
# separate subject
train_eeg1_fft_sub1, train_eeg1_fft_sub2, train_eeg1_fft_sub3 = np.split(train_eeg1_fft, 3, axis=0)
train_eeg2_fft_sub1, train_eeg2_fft_sub2, train_eeg2_fft_sub3 = np.split(train_eeg2_fft, 3, axis=0)
test_eeg1_fft_sub4, test_eeg1_fft_sub5 = np.split(test_eeg1_fft, 2, axis=0)
test_eeg2_fft_sub4, test_eeg2_fft_sub5 = np.split(test_eeg2_fft, 2, axis=0)
# normalization (per freq)
# calc mean and std
mean_train_eeg1_fft_sub1 = calc_mean(train_eeg1_fft_sub1, 'train')
std_train_eeg1_fft_sub1 = calc_std(train_eeg1_fft_sub1, 'train', mean_train_eeg1_fft_sub1)
mean_train_eeg1_fft_sub2 = calc_mean(train_eeg1_fft_sub2, 'train')
std_train_eeg1_fft_sub2 = calc_std(train_eeg1_fft_sub2, 'train', mean_train_eeg1_fft_sub2)
mean_train_eeg1_fft_sub3 = calc_mean(train_eeg1_fft_sub3, 'train')
std_train_eeg1_fft_sub3 = calc_std(train_eeg1_fft_sub3, 'train', mean_train_eeg1_fft_sub3)

mean_train_eeg2_fft_sub1 = calc_mean(train_eeg2_fft_sub1, 'train')
std_train_eeg2_fft_sub1 = calc_std(train_eeg2_fft_sub1, 'train', mean_train_eeg2_fft_sub1)
mean_train_eeg2_fft_sub2 = calc_mean(train_eeg2_fft_sub2, 'train')
std_train_eeg2_fft_sub2 = calc_std(train_eeg2_fft_sub2, 'train', mean_train_eeg2_fft_sub2)
mean_train_eeg2_fft_sub3 = calc_mean(train_eeg2_fft_sub3, 'train')
std_train_eeg2_fft_sub3 = calc_std(train_eeg2_fft_sub3, 'train', mean_train_eeg2_fft_sub3)

mean_test_eeg1_fft_sub4 = calc_mean(test_eeg1_fft_sub4, 'test')
std_test_eeg1_fft_sub4 = calc_std(test_eeg1_fft_sub4, 'test', mean_test_eeg1_fft_sub4)
mean_test_eeg1_fft_sub5 = calc_mean(test_eeg1_fft_sub5, 'test')
std_test_eeg1_fft_sub5 = calc_std(test_eeg1_fft_sub5, 'test', mean_test_eeg1_fft_sub5)

mean_test_eeg2_fft_sub4 = calc_mean(test_eeg2_fft_sub4, 'test')
std_test_eeg2_fft_sub4 = calc_std(test_eeg2_fft_sub4, 'test', mean_test_eeg2_fft_sub4)
mean_test_eeg2_fft_sub5 = calc_mean(test_eeg2_fft_sub5, 'test')
std_test_eeg2_fft_sub4 = calc_std(test_eeg2_fft_sub5, 'test', mean_test_eeg2_fft_sub5)

# normalize (X-mean)/std
train_eeg1_fft_sub1 = normalize(train_eeg1_fft_sub1, mean_train_eeg1_fft_sub1, std_train_eeg1_fft_sub1)
train_eeg1_fft_sub2 = normalize(train_eeg1_fft_sub2, mean_train_eeg1_fft_sub2, std_train_eeg1_fft_sub2)
train_eeg1_fft_sub3 = normalize(train_eeg1_fft_sub3, mean_train_eeg1_fft_sub3, std_train_eeg1_fft_sub3)

train_eeg2_fft_sub1 = normalize(train_eeg2_fft_sub1, mean_train_eeg2_fft_sub1, std_train_eeg2_fft_sub1)
train_eeg2_fft_sub2 = normalize(train_eeg2_fft_sub2, mean_train_eeg2_fft_sub2, std_train_eeg2_fft_sub2)
train_eeg2_fft_sub3 = normalize(train_eeg2_fft_sub3, mean_train_eeg2_fft_sub3, std_train_eeg2_fft_sub3)

test_eeg1_fft_sub4 = normalize(test_eeg1_fft_sub4, mean_test_eeg1_fft_sub4, std_test_eeg1_fft_sub4)
test_eeg1_fft_sub5 = normalize(test_eeg1_fft_sub5, mean_test_eeg1_fft_sub5, std_test_eeg1_fft_sub5)

test_eeg2_fft_sub4 = normalize(test_eeg2_fft_sub4, mean_test_eeg2_fft_sub4, std_test_eeg2_fft_sub4)
test_eeg2_fft_sub5 = normalize(test_eeg2_fft_sub5, mean_test_eeg2_fft_sub5, std_test_eeg2_fft_sub5)
#------------------------------------


## emg preprocessing
#------------------------------------
# summation on frequency & log
#train_emg_fft = np.log(np.sum(train_emg_fft, axis=1)) #(64800, 17)
#test_emg_fft = np.log(np.sum(test_emg_fft, axis=1)) #(43200, 17)
# separate subject
train_emg_fft_sub1, train_emg_fft_sub2, train_emg_fft_sub3 = np.split(train_emg_fft, 3, axis=0)
test_emg_fft_sub4, test_emg_fft_sub5 = np.split(test_emg_fft, 2, axis=0)

# normalization (per freq)
#------------------------------------
