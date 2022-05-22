import pandas as pd

# Names for header
par_names = ['Class',
    'Alcohol',
 	'Malic acid',
 	'Ash',
	'Alcalinity of ash',
 	'Magnesium',
	'Total phenols',
 	'Flavanoids',
 	'Nonflavanoid phenols',
 	'Proanthocyanins',
	'Color intensity',
 	'Hue',
 	'OD280/OD315 of diluted wines',
 	'Proline']

# Reading file with data (setting tab as values separator, encoding to utf-16 and float separator to comma)
data_list = pd.read_csv('./wine.data', sep=',', encoding='utf-8', header=None)#,names=par_names)

# Min-max normalization function definition
def min_max_norm(column):
    return (column - column.min()) / (column.max() - column.min())

# Copy of initial DataFrame
norm_data = data_list.copy()
# apply normalization techniques
for column in norm_data.columns:
    if column !=0:
        norm_data[column] = (norm_data[column] - norm_data[column].min()) / (norm_data[column].max() - norm_data[column].min())

#print(data_list.dtypes)
#print(data_list)
#print(norm_data)
norm_data.to_csv('wine.csv',header=False, index=False)
