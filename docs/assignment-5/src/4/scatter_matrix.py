import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")
plt.show()