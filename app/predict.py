import numpy as np
import json
import os

_model = None
_mapping = None
_inv_mapping = None
_mean = None
_std = None
_top_freqs = None

def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        from config import MODEL_PATH
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def get_mapping():
    global _mapping, _inv_mapping
    if _mapping is None:
        from config import DATA_DIR
        with open(os.path.join(DATA_DIR, 'class_mapping.json'), 'r') as f:
            _mapping = json.load(f)
        _inv_mapping = {v: k for k, v in _mapping.items()}
    return _mapping, _inv_mapping

def get_scaler():
    global _mean, _std, _top_freqs
    if _mean is None:
        from config import DATA_DIR
        _mean = np.load(os.path.join(DATA_DIR, 'feature_mean.npy'))
        _std = np.load(os.path.join(DATA_DIR, 'feature_std.npy'))
        _top_freqs = np.load(os.path.join(DATA_DIR, 'top_freqs.npy'))
    return _mean, _std, _top_freqs

def extract_features(x, top_freqs):
    result = []
    for sample in x:
        a = sample.flatten().astype(np.float32)
        fft = np.abs(np.fft.rfft(a))
        result.append(fft[top_freqs])
    return np.array(result, dtype=np.float32)

def predict(test_x, test_y=None):
    model = get_model()
    _, inv_mapping = get_mapping()
    mean, std, top_freqs = get_scaler()

    x = extract_features(test_x, top_freqs)
    x = (x - mean) / std

    probs = model.predict(x)
    predicted = np.argmax(probs, axis=1)
    predicted_names = [inv_mapping[i] for i in predicted.tolist()]

    result = {'predictions': predicted_names}

    if test_y is not None:
        correct = (predicted == test_y).tolist()
        accuracy = float(np.mean(correct))
        loss = float(-np.mean(np.log(probs[np.arange(len(test_y)), test_y] + 1e-8)))
        result['accuracy'] = accuracy
        result['loss'] = loss
        result['correct'] = correct

    return result
