import numpy as np
import json
import os
import tempfile
import pytest
from app.preprocess import build_class_mapping, encode_labels, preprocess_x, load_and_prepare

def make_fake_y(n=10):
    classes = ['aabbcc' * 5 + 'ClassA', 'ddeeff' * 5 + 'ClassB', '112233' * 5 + 'ClassC']
    return np.array([classes[i % 3] for i in range(n)])

def test_build_class_mapping():
    y = make_fake_y()
    mapping = build_class_mapping(y)
    assert len(mapping) == 3
    assert set(mapping.values()) == {0, 1, 2}

def test_encode_labels():
    y = make_fake_y(6)
    mapping = build_class_mapping(y)
    encoded = encode_labels(y, mapping)
    assert len(encoded) == 6
    assert encoded.dtype in [np.int32, np.int64, np.intp]

def test_preprocess_x_shape():
    x = np.random.randn(5, 80000, 1).astype(np.float32)
    result = preprocess_x(x)
    assert result.shape == (5, 80000, 1)

def test_preprocess_x_normalized():
    x = np.random.randn(5, 80000, 1).astype(np.float32) * 100
    result = preprocess_x(x)
    assert np.abs(result).max() <= 1.0 + 1e-5

def test_preprocess_x_padding():
    x = np.random.randn(3, 40000, 1).astype(np.float32)
    result = preprocess_x(x)
    assert result.shape[1] == 80000

def test_preprocess_x_truncation():
    x = np.random.randn(3, 100000, 1).astype(np.float32)
    result = preprocess_x(x)
    assert result.shape[1] == 80000
