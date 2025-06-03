import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('student\student-merged.csv', sep=';', quoting=1)

print(df.head())

print(df.info())

print("Plotting histogram...")
plt.figure(figsize=(6, 4))
plt.hist(df['G3'], bins=20, edgecolor='black')
plt.title('Histogram of Final Grades (G3)')
plt.xlabel('G3')
plt.ylabel('Count')
plt.show()

print("One-hot encoding...")
df = pd.get_dummies(df, drop_first=True)
print(df.head())

print("Plotting boxplot...")
fig, axes = plt.subplots(1, 3, figsize=(9, 4))
for i, grade in enumerate(['G1', 'G2', 'G3']):
    df.boxplot(column=grade, by='SUBJECT_POR', ax=axes[i], grid=False, widths=0.7)
    axes[i].set_title(f'{grade} by Subject')
    axes[i].set_xlabel(None)
    axes[i].set_xticklabels(['Mathematics', 'Portuguese'])
plt.suptitle('')
plt.tight_layout()
plt.show()

print("Plotting feature correlations...")
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

print("Plotting regression plots...")
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