from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

df['Variety'] = iris.target
df.loc[df['Variety'] == 0, 'Variety'] = 'setosa'
df.loc[df['Variety'] == 1, 'Variety'] = 'versicolor'
df.loc[df['Variety'] == 2, 'Variety'] = 'virginica'
print(df.head())

sns.pairplot(df, hue='Variety')
plt.show()
