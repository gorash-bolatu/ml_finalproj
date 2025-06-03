import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('student\student-merged.csv', sep=';', quoting=1)

print(df.head())

print(df.info())

print("One-hot encoding...")
df = pd.get_dummies(df, drop_first=True)
print(df.head())

print("Splitting into X and y...\t\t\t")
X = df.drop(columns=['G3'])
y = df['G3']

scaler = StandardScaler()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print("Scaled X:")
print(X.head())

df.to_csv('student\student-PROCESSED.csv', sep=';', index=False, quoting=1)