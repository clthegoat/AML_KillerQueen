import pandas as pd
import numpy as np
import os


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


def main():
    print()
    print('***************By Killer Queen***************')
    if not os.path.exists('./data'):
        os.mkdir('./data')
    for file in os.listdir('./data'):
        print(file)
        data = pd.read_csv(os.path.join('./data/', file))
        np.save("./data/"+file[:-4]+'.npy', from_csv_to_ndarray(data))


main()
