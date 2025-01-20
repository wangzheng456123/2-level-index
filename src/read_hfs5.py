import h5py
import numpy as np
import os

def process_dataset(a):
    if a.dtype == 'int32':
        a = a.astype(np.float32)
    return a

def is_data(string):
    return 'train' in string or 'test' in string

def read_hdf5(data: str = "ann"):
    '''if data == "ann", return label is None'''
    if data == "ann":
        data_file_path = "/home/zwang/ag_news-384-euclidean.hdf5"
    elif data == "label":
        data_file_path = "/home/zwang/ag_news-384-euclidean-filter.hdf5"
    else:
        TypeError("The data filter is not supported!")
    f = h5py.File(data_file_path, "r")

    keys = []
    types = []

    dir = "../build/dataset/ag_news/"

    for key in f.keys():
        a = np.array(f[key])
        if is_data(key):
            a = process_dataset(a)
        else:
            fmt = '%.10f'
            if a.dtype == 'int32':
                fmt = '%d'
            np.savetxt(dir + key + '.txt', a, fmt)
        a.tofile(dir + key)
        keys.append(key)
        size = np.array([a.shape[0], a.shape[1]], np.int32)
        size.tofile(dir + key + "_size")
        types.append(a.dtype)
        print(key)

    for type in types:
        print(type)
        

if __name__ == "__main__":
    read_hdf5("label")