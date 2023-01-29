import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from utils import plot_decision_regions

SEED = 19

# Create Dataset
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

fig = plt.figure(figsize=(10,10))
plot_decision_regions(X, y_scaled, classifier=pipe)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
print()