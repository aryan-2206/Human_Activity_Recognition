"""Generate UCI HAR exploratory plots from processed (channel-normalized) signals."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import ROOT, OUT

DEST = ROOT/'results'/'eda'

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    Xtr, Xte = np.load(OUT/'X_train.npy'), np.load(OUT/'X_test.npy')
    ytr, yte = np.load(OUT/'y_train.npy'), np.load(OUT/'y_test.npy')
    labels = json.loads((OUT/'label_names.json').read_text())
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11,5))
    ax.bar(x-.2, np.bincount(ytr,minlength=len(labels)),.4,label='Train')
    ax.bar(x+.2, np.bincount(yte,minlength=len(labels)),.4,label='Test')
    ax.set_xticks(x,labels,rotation=25,ha='right'); ax.set_ylabel('Windows'); ax.set_title('Activity distribution'); ax.legend()
    fig.tight_layout(); fig.savefig(DEST/'class_distribution.png',dpi=160); plt.close(fig)
    t = np.arange(128)/50
    fig, axes = plt.subplots(len(labels),2,figsize=(14,14),sharex=True)
    for i,label in enumerate(labels):
        w = Xtr[np.flatnonzero(ytr==i)[0]]
        for j,(start,title) in enumerate([(0,'Accelerometer'),(3,'Gyroscope')]):
            for k,axis in enumerate('xyz'):
                axes[i,j].plot(t,w[:,start+k],label=axis,linewidth=.9)
            axes[i,j].set_title(f'{label}: {title}'); axes[i,j].legend(loc='upper right',ncol=3)
    for ax in axes[-1]: ax.set_xlabel('Time (s)')
    fig.tight_layout(); fig.savefig(DEST/'signals_by_activity.png',dpi=150); plt.close(fig)
    # Per-window average magnitude, not one pooled magnitude across all windows.
    magnitude = np.linalg.norm(Xtr[:,:,:3],axis=2).mean(axis=1)
    fig,ax = plt.subplots(figsize=(11,5))
    ax.boxplot([magnitude[ytr==i] for i in range(len(labels))],tick_labels=labels,showfliers=False)
    ax.tick_params(axis='x',labelrotation=25); ax.set_ylabel('Mean normalized acceleration magnitude'); ax.set_title('Signal magnitude by activity')
    fig.tight_layout(); fig.savefig(DEST/'magnitude_boxplot.png',dpi=160); plt.close(fig)
    summary = ('Static activities (SITTING, STANDING, LAYING) typically show lower short-term motion variability, '
               'while dynamic activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS) show stronger periodic changes. '
               'Posture-related differences between sitting and standing can be subtle, and body orientation affects the axes. '
               'Plots use train-fitted standardized channels, so magnitudes are in standardized units, not m/s². '
               'The UCI HAR dataset does not include running.')
    (DEST/'summary.txt').write_text(summary+'\n')
    print(summary)
    print('EDA plots saved in',DEST)

if __name__=='__main__': main()
