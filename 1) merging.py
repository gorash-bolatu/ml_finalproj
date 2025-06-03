import pandas as pd

df_mat = pd.read_csv('student\student-mat.csv', sep=';')
df_por = pd.read_csv('student\student-por.csv', sep=';')
df_mat['SUBJECT'] = 'MAT'
df_por['SUBJECT'] = 'POR'
df = pd.concat([df_mat, df_por], ignore_index=True)
df.to_csv('student\student-merged.csv', sep=';', index=False, quoting=1)

print(df.head())
print(df.info())