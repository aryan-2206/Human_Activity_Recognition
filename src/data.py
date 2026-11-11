"""Prepare the official UCI HAR subject-separated train/test inertial windows."""
from pathlib import Path
import json
import zipfile
import urllib.request
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'processed'
URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
CHANNELS = ['total_acc_x', 'total_acc_y', 'total_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

def find_dataset():
    for path in RAW.rglob('UCI HAR Dataset'):
        if (path / 'train' / 'Inertial Signals').is_dir():
            return path
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / 'uci_har.zip'
    if not archive.exists():
        print('Downloading UCI HAR dataset...')
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        # UCI currently serves either the dataset ZIP or an outer ZIP containing it.
        nested = [n for n in z.namelist() if n.lower().endswith('.zip')]
        if nested and not any('UCI HAR Dataset/' in n for n in z.namelist()):
            import io
            with zipfile.ZipFile(io.BytesIO(z.read(nested[0]))) as inner:
                inner.extractall(RAW)
        else:
            z.extractall(RAW)
    for path in RAW.rglob('UCI HAR Dataset'):
        if (path / 'train' / 'Inertial Signals').is_dir():
            return path
    raise FileNotFoundError('Dataset not found after extracting; manually place UCI HAR Dataset in data/raw/')

def read_split(folder, split):
    signal_dir = folder / split / 'Inertial Signals'
    arrays = [np.loadtxt(signal_dir / f'{c}_{split}.txt', dtype=np.float32) for c in CHANNELS]
    X = np.stack(arrays, axis=-1).astype(np.float32)
    y = np.loadtxt(folder / split / f'y_{split}.txt', dtype=np.int64) - 1
    subjects = np.loadtxt(folder / split / f'subject_{split}.txt', dtype=np.int64)
    assert X.ndim == 3 and X.shape[1:] == (128, 6)
    return X, y, subjects

def prepare_data(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    names = ['X_train', 'X_test', 'y_train', 'y_test', 'subjects_train', 'subjects_test']
    if not force and all((OUT / (n + '.npy')).exists() for n in names) and (OUT / 'scaler.joblib').exists():
        return load_data()
    folder = find_dataset()
    X_train, y_train, subjects_train = read_split(folder, 'train')
    X_test, y_test, subjects_test = read_split(folder, 'test')
    scaler = StandardScaler().fit(X_train.reshape(-1, 6))
    X_train = scaler.transform(X_train.reshape(-1, 6)).reshape(X_train.shape).astype(np.float32)
    X_test = scaler.transform(X_test.reshape(-1, 6)).reshape(X_test.shape).astype(np.float32)
    for name, array in zip(names, [X_train, X_test, y_train, y_test, subjects_train, subjects_test]):
        np.save(OUT / (name + '.npy'), array)
    joblib.dump(scaler, OUT / 'scaler.joblib')
    (OUT / 'label_names.json').write_text(json.dumps(LABELS, indent=2))
    return load_data()

def load_data():
    names = ['X_train', 'X_test', 'y_train', 'y_test', 'subjects_train', 'subjects_test']
    arrays = tuple(np.load(OUT / (n + '.npy')) for n in names)
    for split, X, y in [('train', arrays[0], arrays[2]), ('test', arrays[1], arrays[3])]:
        print(f'{split}: X={X.shape}, y={y.shape}, counts={dict(zip(LABELS, np.bincount(y, minlength=6).tolist()))}')
    return arrays

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    prepare_data(force=parser.parse_args().force)
