import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from matplotlib.colors import Normalize
from matplotlib import cm
from helper_module import *

max_depth_list = [5, 10, 20, None]
min_samples_split_list = [10, 30, 50, 70, 100]
total = len(max_depth_list) * len(min_samples_split_list)
best_rmse = float('inf')
best_params = dict()
params = list()
iteration = 0
total_start_time = time.time()

print("\nDecision Tree hyperparameter tuning...")
for min_samples_split in min_samples_split_list:
    for max_depth in max_depth_list:
        iteration += 1
        print(f"Iteration {iteration}/{total}:",
    f"min_samples_split={min_samples_split}",
    f"max_depth={max_depth}",
    sep=" ", end="    \r")
        model = DecisionTreeRegressor(
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            random_state=42
        )
        rmses = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmses.append(rmse)
        avg_rmse = np.mean(rmses)
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = {'min_samples_split': min_samples_split,
                'max_depth': max_depth}
        params.append({'min_samples_split': min_samples_split,
            'max_depth': max_depth,
            'Actual depth': model.get_depth(),
            'rmse': avg_rmse})

print("Best parameters:", best_params)
print(f"Best cross-validated RMSE: {best_rmse:.8f}")
total_time = time.time() - total_start_time
print("Total time:", f"{total_time:.4f}s")
print("Average time per iteration:", f"{(total_time/total):.4f}s")

params_df = pd.DataFrame(params)
params_df.head().fillna("Unlimited")

print("Plotting...", end='\r')
rmse_values = params_df['rmse']
replacement = params_df['max_depth'].dropna().max() + 10
md_values = params_df['max_depth'].fillna(replacement).astype(float)
norm = Normalize(vmin=rmse_values.min(), vmax=rmse_values.max())
cmap = cm.RdYlGn_r # Reversed RdYlGn colormap (Red to Green)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    params_df['min_samples_split'],
    md_values,
    params_df['Actual depth'],
    c=rmse_values, cmap=cmap, norm=norm, s=100)
ax.set_xlabel('min_samples_split', fontsize=14)
ax.set_ylabel('max_depth', fontsize=14)
ax.set_zlabel('(Actual depth)', fontsize=14)
ax.set_xticks(min_samples_split_list)
ax.set_yticks(md_values, ["Unlimited" if (str(i) == "nan") else i for i in params_df['max_depth']])
ax.set_zticks(params_df['Actual depth'])
ax.set_title("Decision Tree: Hyperparameters vs RMSE", pad=10, x=0.7, fontsize=16)
cbar = fig.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('RMSE (root mean square error)', fontsize=16)
plt.show()

print("Decision Tree parameters:", best_params)
train_eval_plot(
    DecisionTreeRegressor(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42
        ),
    "Decision Tree")