import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn import svm

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids

def make_submission(clf, X_test, ids, encoder, name='svm_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    print (y_prob)
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n') 
        for id, probs in zip(ids, y_prob):
            
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))


X, y, encoder, scaler = load_train_data('data/train.csv')

X_test, ids = load_test_data('data/test1.csv', scaler)

num_classes = len(encoder.classes_)
num_features = X.shape[1]

clf = svm.SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0001,
  kernel='rbf', max_iter=-1, probability=True, random_state=None,
  shrinking=True, tol=0.001, verbose=False)

clf.fit(X, y) 


make_submission(clf, X_test, ids, encoder)

