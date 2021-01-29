import pandas as pd

df_dataset = pd.read_excel("C:\\Users\\Admin\\Desktop\\Course_Recommendation\\code\\encoded_data.xlsx")

aggregation_functions = {'Grade': 'mean', 'Attepmpts': 'mean'}
for col in df_dataset.columns:
    if (col != "Course ID" and col != "Attepmpts" and col != "Grade" and col != "Student ID"):
        aggregation_functions[col] = 'sum'
df_new = df_dataset.groupby(df_dataset['Course ID']).aggregate(aggregation_functions)
df_left = df_new['Attepmpts'] + df_new['Grade']
df_left_2 = df_new.drop([ 'Attepmpts', 'Grade'],axis=1)
df_left_2=df_left_2.replace(2,1)
df_new=df_left+df_left_2
print(df_new)
writer = pd.ExcelWriter('recommend.xlsx', engine='xlsxwriter')
df_new.to_excel(writer, sheet_name='Sheet1')
writer.save()
