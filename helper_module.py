import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
import time

df = pd.read_csv('student\student-PROCESSED.csv', sep=';', quoting=1)
X = df.drop(columns=['G3'])
y = df['G3']

cv = KFold(n_splits=5, shuffle=True, random_state=420)

def train_eval_plot(model, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training...", end='\r')
    training_start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    training_time = time.time() - training_start
    print("Training time:", f"{training_time:.4f}s")

    eval_start = time.time()
    print("Cross-validating RMSE...", end="\r")
    neg_mse = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-neg_mse)
    print("Cross-validated RMSE:", rmse_scores.mean())
    print("Cross-validating MAE...", end="\r")
    neg_mae = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    mae_scores = -neg_mae
    print("Cross-validated MAE:", mae_scores.mean())
    print("Cross-validating R-squared...", end="\r")
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print("Cross-validated R-squared:", r2_scores.mean())
    evaulation_time = time.time() - eval_start
    print("Evaulation time:", f"{evaulation_time:.4f}s")

    plt.figure(figsize=(5, 4))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual G3')
    plt.ylabel('Predicted G3')
    plt.title(f'{model_name}: Predicted vs. Actual Grades')
    plt.grid(True)
    plt.tight_layout()
    plt.show()