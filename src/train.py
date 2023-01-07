import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

DATA_INPUT = '../data/input/synthetic_data.txt'
SAMPLE_SIZE = 2500
SEED = 1993

def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

#df = pd.read_csv(DATA_INPUT)

# Create Dataset
np.random.seed(SEED)
X = np.random.randn(250, 2)
y = np.logical_xor(X[:, 0] > 0,
                       X[:, 1] > 0)
y = np.where(y, 1, -1)

#X = df.iloc[:,:-1]
#y = df['y']

# select a sample for a better control of the research and wall time
#X = X[:SAMPLE_SIZE]
#y = y[:SAMPLE_SIZE]

# scaling the labels to -1, 1 important for the SVM and
# the definition of the hinge loss
#y_scaled = 2 * (y - 0.5)
y_scaled = y

# split train and test samples
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_scaled,
    test_size = 0.2, 
    stratify = y_scaled,
    random_state = SEED)

# split train into train and validation samples
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size = 0.25,
    stratify = y_train,
    random_state=SEED
)

""" Classic Machine Learning Pipeline """
# Pipeline definition: normalize the inputs, MinMaxScaled and model
pipe = Pipeline(
    [
        ('normalaizer', StandardScaler()),
        ('scaler', MinMaxScaler((-1,1))),
        ('model', SVC(kernel='linear'))
    ]
)

# fiting
pipe.fit(X_train, y_train)

'''
score = pipe.score(X_val, y_val)
print(f"ML - Classification validation score: {score}")

# inference
y_pred = pipe.predict(X_test)
report = classification_report(y_test,y_pred)
print("ML - Testing report:")
print(report)
conf_matrix = confusion_matrix(y_test,y_pred)
print("ML - Confussion Matrix:")
print(conf_matrix) 
'''
fig = plt.figure(figsize=(10,10))
plot_decision_regions(X, y_scaled, classifier=pipe)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel

import logging
from qiskit_nature import logging as nature_logging
nature_logging.set_levels_for_names(
    {"qiskit_nature": logging.DEBUG, "qiskit": logging.DEBUG})

""" CQ - Quantum Machine Learning Pipeline """
N_DIM = 2
N_SHOTS = 256

# Define feature_map
feature_map = ZZFeatureMap(
    feature_dimension=N_DIM, 
    reps=2,
    entanglement='linear')

# Define the backend
backend = QuantumInstance(
    BasicAer.get_backend("qasm_simulator"), 
    shots=N_SHOTS, 
    seed_simulator=SEED, 
    seed_transpiler=SEED
)

# Define the kernel
kernel = QuantumKernel(
    feature_map=feature_map, 
    quantum_instance=backend)

# QML pipeline
pipe_qml = Pipeline(
    [
        ('pca', PCA(n_components=N_DIM)),
        ('normalaizer', StandardScaler()),
        ('scaler', MinMaxScaler((-1,1))),
        ('model', SVC(kernel=kernel.evaluate))
    ]
)

# fiting
pipe_qml.fit(X_train, y_train)
'''
score = pipe_qml.score(X_val, y_val)
print(f"QML - Classification validation score: {score}")

# inference
y_pred = pipe_qml.predict(X_test)
report = classification_report(y_test,y_pred)
print("QML - Testing report:")
print(report)
conf_matrix = confusion_matrix(y_test,y_pred)
print("QML - Confussion Matrix:")
print(conf_matrix)
'''

fig1 = plt.figure(figsize=(10,10))
plot_decision_regions(X, y_scaled, classifier=pipe_qml)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()