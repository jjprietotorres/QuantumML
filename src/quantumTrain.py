import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel

import logging
from qiskit_nature import logging as nature_logging

from utils import plot_decision_regions

nature_logging.set_levels_for_names(
    {"qiskit_nature": logging.DEBUG, "qiskit": logging.DEBUG})

SEED = 19

# Create non-linear Dataset
np.random.seed(SEED)
X = np.random.randn(250, 2)
y = np.logical_xor(X[:, 0] > 0,
                       X[:, 1] > 0)
y = np.where(y, 1, -1)
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