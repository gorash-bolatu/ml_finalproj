import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matplotlib.colors import Normalize
from matplotlib import cm
from helper_module import *

n_estimators_list = [10, 30, 60, 100]
max_depth_list = [5, 10, 20, None]
min_samples_split_list = [10, 30, 50, 70, 100]
total = len(n_estimators_list) * len(max_depth_list) * len(min_samples_split_list)
best_rmse = float('inf')
best_params = dict()
params = list()
iteration = 0
total_start_time = time.time()

print("\nRandom Forest hyperparameter tuning...")
for n_estimators in n_estimators_list:
    for min_samples_split in min_samples_split_list:
        for max_depth in max_depth_list:
            iteration += 1
            print(f"Iteration {iteration}/{total}:",
                    f"n_estimators={n_estimators}",
                    f"min_samples_split={min_samples_split}",
                    f"max_depth={max_depth}",
                    sep=" ", end="       \r")
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                max_depth=max_depth,
                n_jobs=-1,
                random_state=42
            )
            iteration_start_time = time.time()
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
                best_params = {'n_estimators': n_estimators,
                                'min_samples_split': min_samples_split,
                                'max_depth': max_depth}
            iteration_time = time.time() - iteration_start_time
            params.append({'n_estimators': n_estimators,
                            'min_samples_split': min_samples_split,
                            'max_depth': max_depth,
                            'rmse': avg_rmse,
                            'time': iteration_time})

print("Best parameters:", best_params)
print(f"Best cross-validated RMSE: {best_rmse:.8f}")
total_time = time.time() - total_start_time
print("Total time:", f"{total_time:.4f}s")
print("Average time per iteration:",
    f"{(total_time/total):.4f}s;",
    "Mean:",
    f"{np.mean([i['time'] for i in params]):.4f}s")

params_df = pd.DataFrame(params)
params_df.head().fillna("Unlimited")

print("Plotting...", end='\r')
valid_depths = params_df['max_depth'].dropna()
replacement = valid_depths.max() + 10
md_values = params_df['max_depth'].fillna(replacement)
depth_ticks = sorted(md_values.unique())
depth_labels = ["Unlimited" if pd.isna(orig) else str(int(orig))
                for orig in params_df['max_depth'].unique()]
norm_rmse = Normalize(vmin=params_df['rmse'].min(), vmax=params_df['rmse'].max())
norm_time = Normalize(vmin=params_df['time'].min(), vmax=params_df['time'].max())
cmap = cm.RdYlGn_r

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(18, 8))

sc1 = ax1.scatter(
    params_df['n_estimators'], md_values, params_df['min_samples_split'],
    c=params_df['rmse'], cmap=cmap, norm=norm_rmse, s=100
)
ax1.set_xlabel('n_estimators', fontsize=12)
ax1.set_ylabel('max_depth', fontsize=12)
ax1.set_zlabel('min_samples_split', fontsize=12)
ax1.set_xticks(params_df['n_estimators'])
ax1.set_yticks(depth_ticks)
ax1.set_yticklabels(depth_labels)
ax1.set_title("Hyperparameters vs RMSE", pad=10, x=0.5, fontsize=14)
cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
cbar1.set_label('RMSE (root mean squared error)', fontsize=12)

sc2 = ax2.scatter(
    params_df['n_estimators'], md_values, params_df['min_samples_split'],
    c=params_df['time'], cmap=cmap, norm=norm_time, s=100
)
ax2.set_xlabel('n_estimators', fontsize=12)
ax2.set_ylabel('max_depth', fontsize=12)
ax2.set_zlabel('min_samples_split', fontsize=12)
ax2.set_xticks(params_df['n_estimators'])
ax2.set_yticks(depth_ticks)
ax2.set_yticklabels(depth_labels)
ax2.set_title("Hyperparameters vs Training & Evaluation Time", pad=10, x=0.5, fontsize=14)
cbar2 = fig.colorbar(sc2, ax=ax2, pad=0.1)
cbar2.set_label('Time (s)', fontsize=12)

fig.subplots_adjust(wspace=0.1)
fig.suptitle("Random Forest Hyperparameter Tuning Results", fontsize=21, y=0.98)
plt.show()

print("Random Forest parameters:", best_params)
train_eval_plot(
    RandomForestRegressor(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42),
    "Random Forest")
