"""Time-domain and FFT features; works for a single window or batches."""
from pathlib import Path
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from src.data import ROOT, OUT

CHANNELS = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']
STATS = ['mean','std','min','max','median','mad','iqr','energy','skewness','kurtosis','dominant_fft_hz','spectral_energy']
MAG_STATS = ['mean','std','min','max','energy']
FEATURE_NAMES = [f'{ch}_{stat}' for ch in CHANNELS for stat in STATS] + [f'{kind}_magnitude_{stat}' for kind in ('acc','gyro') for stat in MAG_STATS] + [f'acc_corr_{a}_{b}' for a,b in [(0,1),(0,2),(1,2)]]

def extract_batch(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != (128, 6):
        raise ValueError('Expected X shape (N,128,6)')
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    centered = X - mean[:, None, :]
    safe_std = np.maximum(std, 1e-9)
    fft = np.fft.rfft(centered, axis=1)
    power = np.abs(fft) ** 2
    freq = np.fft.rfftfreq(128, d=1/50)
    # Exclude DC; identify the strongest non-zero frequency bin.
    dominant = freq[1:][np.argmax(power[:, 1:, :], axis=1)]
    channel_features = [
        mean, std, X.min(axis=1), X.max(axis=1), np.median(X,axis=1),
        np.mean(np.abs(centered),axis=1),
        np.percentile(X,75,axis=1)-np.percentile(X,25,axis=1),
        np.mean(X**2,axis=1),
        np.mean((centered/safe_std[:,None,:])**3,axis=1),
        np.mean((centered/safe_std[:,None,:])**4,axis=1)-3,
        dominant, np.mean(power,axis=1)
    ]
    result = np.stack(channel_features,axis=-1).reshape(len(X), -1)
    mags = []
    for sl in [slice(0,3),slice(3,6)]:
        m = np.linalg.norm(X[:,:,sl],axis=2)
        mags.append(np.stack([m.mean(1),m.std(1),m.min(1),m.max(1),np.mean(m*m,axis=1)],axis=1))
    corrs = []
    for a,b in [(0,1),(0,2),(1,2)]:
        covariance = np.mean(centered[:,:,a]*centered[:,:,b],axis=1)
        corrs.append(np.clip(covariance / np.maximum(std[:,a]*std[:,b],1e-9),-1,1))
    result = np.column_stack([result, *mags, *corrs])
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def extract_features(window):
    window = np.asarray(window)
    if window.shape != (128,6):
        raise ValueError('Expected window shape (128,6)')
    return extract_batch(window[None,:,:])[0], FEATURE_NAMES.copy()

def main():
    t = time.perf_counter()
    X_train = np.load(OUT/'X_train.npy')
    X_test = np.load(OUT/'X_test.npy')
    F_train_raw = extract_batch(X_train)
    F_test_raw = extract_batch(X_test)
    scaler = StandardScaler().fit(F_train_raw)
    np.save(OUT/'F_train.npy', scaler.transform(F_train_raw).astype(np.float32))
    np.save(OUT/'F_test.npy', scaler.transform(F_test_raw).astype(np.float32))
    joblib.dump(scaler, OUT/'feature_scaler.joblib')
    (OUT/'feature_names.json').write_text(json.dumps(FEATURE_NAMES,indent=2))
    print(f'Feature extraction complete: {F_train_raw.shape[1]} features; train {F_train_raw.shape}, test {F_test_raw.shape}; {time.perf_counter()-t:.2f}s')

if __name__ == '__main__':
    main()
