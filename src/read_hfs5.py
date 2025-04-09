import h5py
import numpy as np
import os

def process_dataset(a):
    if a.dtype == 'int32':
        a = a.astype(np.float32)
    if a.dtype == 'int64':
        a = a.astype(np.float32)
    if a.dtype == 'float64':
        a = a.astype(np.float32)
    return a

def is_data(string):
    return 'train' in string or 'test' in string

def is_label(string):
    return 'label' in string

def get_data_file_path(data: str) -> str:
    """Get the file path based on the data type."""
    if data == "ann":
        return "/home/zwang/sift10m-6filter-6a.hdf5"
    elif data == "label":
        return "/home/zwang/sift10m-6filter-6a.hdf5"
    else:
        raise TypeError("The data filter is not supported!")

def process_key_data(f, key, dir_path: str):
    """Process the data for a given key."""
    a = np.array(f[key])
    if is_data(key):
        a = process_dataset(a)
        if is_label(key):
            rows_to_write = a[:100]
            print(rows_to_write)
            #np.savetxt(dir_path + key + ".txt", rows_to_write, delimiter=',', fmt='%.6f')
    else:
        fmt = '%.10f'
        if a.dtype == 'int32':
            fmt = '%d'
        if a.dtype == 'int64':
            fmt = '%d'
            a = a.astype(np.int32)
        np.savetxt(dir_path + key + '.txt', a, fmt)

    a.tofile(dir_path + key)
    size = np.array([a.shape[0], a.shape[1]], np.int32)
    size.tofile(dir_path + key + "_size")
    return key, a.dtype

def process_neighbors(f):
    """Process neighbors, train_labels, and test_labels and compute min/max of dif vectors."""
    neighbors = np.array(f["neighbors"])
    train_labels = np.array(f["train_label"])
    test_labels = np.array(f["test_label"])

    dif_list = [] 

    for i in range(neighbors.shape[0]):
        test_label_row = test_labels[i]
        for idx in neighbors[i]:
            train_data_from_neighbors = train_labels[idx]
            dif = train_data_from_neighbors - test_label_row
            print(train_data_from_neighbors)
            if dif[1] > 30 or dif[1] < -30 or dif[0] > 0:
                continue
            dif_list.append(dif)  
            print(f"dif : {dif}")

    if dif_list:
        dif_array = np.array(dif_list)  
        max_values = np.max(dif_array, axis=0)  
        min_values = np.min(dif_array, axis=0)  

        print(f"Dif vector max values (per dimension): {max_values}")
        print(f"Dif vector min values (per dimension): {min_values}")
    else:
        print("No dif vectors found.")

def read_hdf5(data: str = "ann"):
    """Read and process the HDF5 file."""
    data_file_path = get_data_file_path(data)
    f = h5py.File(data_file_path, "r")

    keys = []
    types = []

    dir_path = "../build/dataset/sift10m/"

    for key in f.keys():
        key, dtype = process_key_data(f, key, dir_path)
        keys.append(key)
        types.append(dtype)
        print(key)

    for dtype in types:
        print(dtype)

    # if {"neighbors", "train_label", "test_label"}.issubset(f.keys()):
    #    process_neighbors(f)
        

if __name__ == "__main__":
    read_hdf5("label")