import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
import time
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matplotlib.colors import Normalize
from matplotlib import cm

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

if __name__ == "__main__":
    df_mat = pd.read_csv('student\student-mat.csv', sep=';')
    df_por = pd.read_csv('student\student-por.csv', sep=';')
    df_mat['SUBJECT'] = 'MAT'
    df_por['SUBJECT'] = 'POR'
    df = pd.concat([df_mat, df_por], ignore_index=True)
    df.to_csv('student\student-merged.csv', sep=';', index=False, quoting=1)

    print(df.head())

    print(df.info())

    print("Missing values: ", df.isna().sum().sum())

    print("One-hot encoding...")
    df = pd.get_dummies(df, drop_first=True)
    print(df.head())

    print("Plotting histogram...", end='\r')
    plt.figure(figsize=(6, 4))
    plt.hist(df['G3'], bins=20, edgecolor='black')
    plt.title('Histogram of Final Grades (G3)')
    plt.xlabel('G3')
    plt.ylabel('Count')
    plt.show()

    print("Plotting boxplot...", end='\r')
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    for i, grade in enumerate(['G1', 'G2', 'G3']):
        df.boxplot(column=grade, by='SUBJECT_POR', ax=axes[i], grid=False, widths=0.7)
        axes[i].set_title(f'{grade} by Subject')
        axes[i].set_xlabel(None)
        axes[i].set_xticklabels(['Mathematics', 'Portuguese'])
    plt.suptitle('')
    plt.tight_layout()
    plt.show()

    print("Plotting feature correlations...", end='\r')
    correlations = df.corr()['G3'].abs().drop('G3').sort_values(ascending=True)
    plt.figure(figsize=(9, 7))
    bars = plt.barh(correlations.index, correlations.values)
    plt.title('Feature Correlation with G3')
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2.5,
                f'{width:.3f}', va='center', fontsize=10)
    ax = plt.gca()
    new_labels = [label.get_text().split('_', 1)[0] for label in ax.get_yticklabels()]
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(new_labels)
    ax.grid(axis='both', alpha=0.5)
    plt.ylim(bars[0].get_y() - 0.5, bars[-1].get_y() + bars[-1].get_height() + 0.5)
    plt.tight_layout()
    plt.show()

    print("Plotting regression plots...", end='\r')
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(ax=axes[0], data=df, x='G1', y='G3', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[0].set_xlabel('G1 (1st Period Grade)', fontsize=16)
    axes[0].set_ylabel('G3 (Final Grade)', fontsize=16)
    sns.regplot(ax=axes[1], data=df, x='G2', y='G3', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[1].set_xlabel('G2 (2nd Period Grade)', fontsize=16)
    axes[1].set_ylabel('G3 (Final Grade)', fontsize=16)
    for i in axes:
        i.set_xticks(range(df['G3'].min(), df['G3'].max()+1))
    plt.tight_layout()
    plt.show()

    print("Splitting into X and y...\t\t\t")
    X = df.drop(columns=['G3'])
    y = df['G3']

    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    print("Scaled X:")
    print(X.head())

    print("\nLinear Regression")
    train_eval_plot(LinearRegression(), "Linear Regression")

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
    train_eval_plot(DecisionTreeRegressor(max_depth=best_params['max_depth'],
                                min_samples_split=best_params['min_samples_split'],
                                random_state=42),
        "Decision Tree")

    n_estimators_list = [10, 30, 60, 100]
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
    train_eval_plot(RandomForestRegressor(max_depth=best_params['max_depth'],
                                min_samples_split=best_params['min_samples_split'],
                                random_state=42),
        "Random Forest")
