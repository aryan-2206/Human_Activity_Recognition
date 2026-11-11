"""KNN baseline and subject-wise CV tuning on train-only standardized features."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from src.data import ROOT, OUT

RESULTS = ROOT/'results'

def save_evaluation(name,y_true,y_pred,labels,train_time_s,infer_ms_per_sample):
    """Fallback compatible with teammate's shared evaluate_and_save contract."""
    RESULTS.mkdir(exist_ok=True)
    cm = confusion_matrix(y_true,y_pred,labels=np.arange(len(labels)))
    payload = dict(accuracy=float(accuracy_score(y_true,y_pred)),
                   macro_f1=float(f1_score(y_true,y_pred,average='macro',zero_division=0)),
                   per_class_report=classification_report(y_true,y_pred,labels=np.arange(len(labels)),target_names=labels,output_dict=True,zero_division=0),
                   confusion_matrix=cm.tolist(),train_time_s=float(train_time_s),infer_ms_per_sample=float(infer_ms_per_sample))
    (RESULTS/f'{name}_metrics.json').write_text(json.dumps(payload,indent=2))
    fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(cm,cmap='Blues'); fig.colorbar(im,ax=ax)
    ax.set_xticks(range(len(labels)),labels,rotation=40,ha='right');ax.set_yticks(range(len(labels)),labels)
    ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title('KNN confusion matrix')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(RESULTS/f'{name}_confusion.png',dpi=160);plt.close(fig)
    return payload

def main():
    RESULTS.mkdir(exist_ok=True);(ROOT/'models').mkdir(exist_ok=True)
    Ftr=np.load(OUT/'F_train.npy');Fte=np.load(OUT/'F_test.npy')
    ytr=np.load(OUT/'y_train.npy');yte=np.load(OUT/'y_test.npy')
    groups=np.load(OUT/'subjects_train.npy')
    labels=json.loads((OUT/'label_names.json').read_text())
    baseline=KNeighborsClassifier(n_neighbors=5)
    start=time.perf_counter();baseline.fit(Ftr,ytr)
    print(f'Baseline k=5 test accuracy: {accuracy_score(yte,baseline.predict(Fte)):.4f}')
    # Select hyperparameters exclusively on train subjects; do not use test set for selection.
    grid={'n_neighbors':[3,5,7,9,11], 'weights':['uniform','distance'], 'metric':['euclidean','manhattan']}
    search=GridSearchCV(KNeighborsClassifier(),grid,cv=GroupKFold(n_splits=3),scoring='accuracy',n_jobs=-1,refit=True,verbose=1)
    start=time.perf_counter();search.fit(Ftr,ytr,groups=groups)
    train_time=time.perf_counter()-start
    model=search.best_estimator_
    joblib.dump(model,ROOT/'models'/'knn.joblib')
    (RESULTS/'knn_best_params.json').write_text(json.dumps({'best_params':search.best_params_,'best_cv_accuracy':float(search.best_score_)},indent=2))
    # Timing includes predict_proba, as used by the application.
    batch=Fte[:min(200,len(Fte))]
    model.predict_proba(batch)
    start=time.perf_counter();model.predict_proba(batch)
    infer_ms=(time.perf_counter()-start)*1000/len(batch)
    pred=model.predict(Fte)
    # Prefer Dhairya's shared helper when present, otherwise write identical schema.
    try:
        from src.evaluate import evaluate_and_save
    except ImportError:
        payload=save_evaluation('knn',yte,pred,labels,train_time,infer_ms)
    else:
        payload=evaluate_and_save('knn',yte,pred,labels,train_time,infer_ms)
    results=search.cv_results_
    fig,ax=plt.subplots(figsize=(8,5))
    for metric in ['euclidean','manhattan']:
        for weights in ['uniform','distance']:
            pairs=sorted((int(p['n_neighbors']),float(s)) for p,s in zip(results['params'],results['mean_test_score']) if p['metric']==metric and p['weights']==weights)
            ax.plot([a for a,b in pairs],[b for a,b in pairs],marker='o',label=f'{metric}, {weights}')
    ax.set_xlabel('k neighbors');ax.set_ylabel('Mean GroupKFold validation accuracy');ax.set_title('KNN tuning curve');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(RESULTS/'knn_k_curve.png',dpi=160);plt.close(fig)
    print('Best parameters:',search.best_params_,'CV accuracy:',round(search.best_score_,4))
    print('Test accuracy:',round(accuracy_score(yte,pred),4),'Train+tune seconds:',round(train_time,2))

if __name__=='__main__': main()
