import numpy as np
import json
import os

TARGET_LEN = 80000

def build_class_mapping(train_y):
    names = sorted(set(s[32:] for s in train_y))
    return {name: i for i, name in enumerate(names)}

def encode_labels(y, mapping):
    return np.array([mapping[s[32:]] for s in y])

def preprocess_x(x):
    x = x.astype(np.float32)
    max_vals = np.max(np.abs(x), axis=(1, 2), keepdims=True)
    max_vals[max_vals == 0] = 1
    x = x / max_vals
    if x.shape[1] < TARGET_LEN:
        pad = TARGET_LEN - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad), (0, 0)))
    else:
        x = x[:, :TARGET_LEN, :]
    return x

def load_and_prepare(data_dir):
    train_x = np.load(os.path.join(data_dir, 'train_x.npy'), allow_pickle=True)
    train_y = np.load(os.path.join(data_dir, 'train_y.npy'), allow_pickle=True)
    valid_x = np.load(os.path.join(data_dir, 'valid_x.npy'), allow_pickle=True)
    valid_y = np.load(os.path.join(data_dir, 'valid_y.npy'), allow_pickle=True)

    mapping = build_class_mapping(train_y)

    mapping_path = os.path.join(data_dir, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    train_x = preprocess_x(train_x)
    valid_x = preprocess_x(valid_x)
    train_y = encode_labels(train_y, mapping)
    valid_y = encode_labels(valid_y, mapping)

    return train_x, train_y, valid_x, valid_y, mapping
