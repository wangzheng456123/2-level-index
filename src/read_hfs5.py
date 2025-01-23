import h5py
import numpy as np
import os

def process_dataset(a):
    if a.dtype == 'int32':
        a = a.astype(np.float32)
    return a

def is_data(string):
    return 'train' in string or 'test' in string

def is_label(string):
    return 'label' in string

def get_data_file_path(data: str) -> str:
    """Get the file path based on the data type."""
    if data == "ann":
        return "/home/zwang/ag_news-384-euclidean.hdf5"
    elif data == "label":
        return "/home/zwang/app_reviews-384-euclidean-filter.hdf5"
    else:
        raise TypeError("The data filter is not supported!")

def process_key_data(f, key, dir_path: str):
    """Process the data for a given key."""
    a = np.array(f[key])
    if is_data(key):
        a = process_dataset(a)
        if is_label(key):
            rows_to_write = a[:100]
            # np.savetxt(dir_path + key + ".txt", rows_to_write, delimiter=',', fmt='%.6f')
    else:
        fmt = '%.10f'
        if a.dtype == 'int32':
            fmt = '%d'
        # np.savetxt(dir_path + key + '.txt', a, fmt)

    # a.tofile(dir_path + key)
    size = np.array([a.shape[0], a.shape[1]], np.int32)
    # size.tofile(dir_path + key + "_size")
    return key, a.dtype

def process_neighbors(f):
    """Process neighbors, train_labels, and test_labels."""
    neighbors = np.array(f["neighbors"])
    train_labels = np.array(f["train_label"])
    test_labels = np.array(f["test_label"])

    for i in range(neighbors.shape[0]):
        for idx in neighbors[i]:
            train_data_from_neighbors = train_labels[idx]
            print(f"Neighbors (row {i}): {idx}")
            print(f"Train labels from neighbors (row {i}): {train_data_from_neighbors}")

            test_label_row = test_labels[i]
            print(f"Test label (row {i}): {test_label_row}")

            if np.all(train_data_from_neighbors == test_label_row):
                print(f"Row {i}: Train labels match the test label!")
            else:
                print(f"Row {i}: Train labels do not match the test label!")

def read_hdf5(data: str = "ann"):
    """Read and process the HDF5 file."""
    data_file_path = get_data_file_path(data)
    f = h5py.File(data_file_path, "r")

    keys = []
    types = []

    dir_path = "../build/dataset/app_review/"

    for key in f.keys():
        key, dtype = process_key_data(f, key, dir_path)
        keys.append(key)
        types.append(dtype)
        print(key)

    for dtype in types:
        print(dtype)

    if {"neighbors", "train_label", "test_label"}.issubset(f.keys()):
        process_neighbors(f)
        

if __name__ == "__main__":
    read_hdf5("label")