import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from app.predict import extract_features

def test_extract_features_shape():
    x = np.random.randn(5, 80000, 1).astype(np.float32)
    top_freqs = np.arange(500)
    result = extract_features(x, top_freqs)
    assert result.shape == (5, 500)

def test_extract_features_dtype():
    x = np.random.randn(3, 80000, 1).astype(np.float32)
    top_freqs = np.arange(100)
    result = extract_features(x, top_freqs)
    assert result.dtype == np.float32

def test_extract_features_values_finite():
    x = np.random.randn(4, 80000, 1).astype(np.float32)
    top_freqs = np.arange(200)
    result = extract_features(x, top_freqs)
    assert np.all(np.isfinite(result))

def test_predict_output_structure():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.eye(20)[:5] * 0.9 + 0.005

    mock_mapping = {str(i): i for i in range(20)}
    mock_inv = {i: str(i) for i in range(20)}
    mock_mean = np.zeros(500)
    mock_std = np.ones(500)
    mock_top_freqs = np.arange(500)

    with patch('app.predict.get_model', return_value=mock_model), \
         patch('app.predict.get_mapping', return_value=(mock_mapping, mock_inv)), \
         patch('app.predict.get_scaler', return_value=(mock_mean, mock_std, mock_top_freqs)):

        from app.predict import predict
        x = np.random.randn(5, 80000, 1).astype(np.float32)
        result = predict(x)

        assert 'predictions' in result
        assert len(result['predictions']) == 5

def test_predict_with_labels():
    mock_model = MagicMock()
    probs = np.zeros((5, 20))
    probs[np.arange(5), np.arange(5)] = 0.9
    mock_model.predict.return_value = probs

    mock_mapping = {str(i): i for i in range(20)}
    mock_inv = {i: str(i) for i in range(20)}
    mock_mean = np.zeros(500)
    mock_std = np.ones(500)
    mock_top_freqs = np.arange(500)

    with patch('app.predict.get_model', return_value=mock_model), \
         patch('app.predict.get_mapping', return_value=(mock_mapping, mock_inv)), \
         patch('app.predict.get_scaler', return_value=(mock_mean, mock_std, mock_top_freqs)):

        from app.predict import predict
        x = np.random.randn(5, 80000, 1).astype(np.float32)
        y = np.arange(5)
        result = predict(x, y)

        assert 'accuracy' in result
        assert 'loss' in result
        assert 'correct' in result
        assert result['accuracy'] == 1.0
