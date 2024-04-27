import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.layers import SimpleRNN, Dense,Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


data=pd.read_csv("C:/Users/HP/Downloads/Breast Cancer Data.csv")
drop_list = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean',
             'radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst',
             'concavity_worst','compactness_se','concave points_se','texture_worst','area_worst','id','diagnosis','Unnamed: 32']
x = data.drop(drop_list ,axis = 1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=54)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0),
    'Logistic Regression': LogisticRegression(),
    'k-NN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'MLP': MLPClassifier(max_iter=1000),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'Perceptron': Perceptron(),
    'SVM': SVC(kernel='linear'),
    'Gaussian Processes': GaussianProcessClassifier(1.0 * RBF(1.0))
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)

# Evaluate models
for name, model in models.items():
    ac = accuracy_score(y_test, model.predict(X_test)) * 100
    print(f'Accuracy for {name}: {ac:.2f}%')



X = data.drop(drop_list, axis=1).values.reshape(-1, 5, 5)  # Reshape for RNN
y = data['diagnosis']
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=54)

# Build RNN model
model_rnn = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(5, 5)),
    Dense(1, activation='sigmoid')
])

model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train RNN model
model_rnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate RNN model
y_pred_rnn = (model_rnn.predict(X_test) > 0.5).astype("int32")
accuracy_rnn = accuracy_score(y_test, y_pred_rnn)
print(f"Accuracy for RNN: {accuracy_rnn * 100:.2f}%")

# Build CNN model
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(5, 5, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN model
model_cnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate CNN model
y_pred_cnn = (model_cnn.predict(X_test) > 0.5).astype("int32")
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)
print(f"Accuracy for CNN: {accuracy_cnn * 100:.2f}%")