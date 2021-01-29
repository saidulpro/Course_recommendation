import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl

df_dataset = pd.read_excel("C:\\Users\\Admin\\Desktop\\Course_Recommendation\\code\\data\\Dataset1.xlsx")

df_dataset.head()
cat_df_dataset = df_dataset.select_dtypes(include=['object']).copy()

i = 0
j = 0
for skill in df_dataset['Skills'].values:
    if not isinstance(skill, int):
        df_dataset['Skills'].values[i] = df_dataset['Skills'].values[i].split(' ')
        for l in df_dataset['Skills'].values[i]:
            df_dataset['Skills'].values[i][j] = int(df_dataset['Skills'].values[i][j])
            j = j + 1
    i = i + 1
    j = 0

i = 0
j = 0
for relT in df_dataset['Related Topics'].values:
    if not isinstance(relT, int):
        df_dataset['Related Topics'].values[i] = df_dataset['Related Topics'].values[i].split(' ')
        for l in df_dataset['Related Topics'].values[i]:
            df_dataset['Related Topics'].values[i][j] = int(df_dataset['Related Topics'].values[i][j])
            j = j + 1
    i = i + 1
    j = 0

i = 0
j = 0
for skill in df_dataset['Related Subjects'].values:
    if not isinstance(skill, int):
        df_dataset['Related Subjects'].values[i] = df_dataset['Related Subjects'].values[i].split(' ')
        for l in df_dataset['Related Subjects'].values[i]:
            df_dataset['Related Subjects'].values[i][j] = int(df_dataset['Related Subjects'].values[i][j])
            j = j + 1
    i = i + 1
    j = 0

related_topics = pd.get_dummies(df_dataset['Related Topics'].apply(pd.Series).stack(),prefix='Related Topic').sum(level=0)
df_dataset=df_dataset.drop(['Related Topics'],axis=1)
df_dataset=df_dataset.join(related_topics)

related_subjects = pd.get_dummies(df_dataset['Related Subjects'].apply(pd.Series).stack(),prefix='Related Subject').sum(level=0)
df_dataset=df_dataset.drop(['Related Subjects'],axis=1)
df_dataset=df_dataset.join(related_subjects)


skills = pd.get_dummies(df_dataset['Skills'].apply(pd.Series).stack(), prefix='Skill').sum(level=0)
df_dataset = df_dataset.drop(['Skills'], axis=1)
df_dataset = df_dataset.join(skills)


print(df_dataset)

sns.countplot(df_dataset['Course ID'],label="Count")
plt.show()
aggregation_functions = {'Grade': 'mean', 'Attepmpts': 'mean'}
df_new = df_dataset.groupby(df_dataset['Course ID']).aggregate(aggregation_functions).reset_index()
sns.barplot(df_new['Course ID'],df_new['Grade'])
plt.show()
sns.barplot(df_new['Course ID'],df_new['Attepmpts'])
plt.show()

writer = pd.ExcelWriter('encoded_data.xlsx', engine='xlsxwriter')
df_dataset.to_excel(writer, sheet_name='Sheet1')
writer.save()
