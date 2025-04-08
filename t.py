import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample DataFrame

df = pd.read_csv('/home/cuong/PycharmProjects/memories/e_n.csv')

# Calculate correlation coefficient
correlation = df['n_token'].corr(df['eta'])

# Create visualization
plt.figure(figsize=(10, 6))

# Scatter plot with regression line
sns.regplot(
    x='n_token',
    y='eta',
    data=df,
    line_kws={'color': 'red'},
    scatter_kws={'alpha': 0.6}
)

# Annotation and labels
plt.title(f"Feature Correlation (r = {correlation:.2f})", fontsize=14)
plt.xlabel('n_token A', fontsize=12)
plt.ylabel('eta B', fontsize=12)
plt.grid(alpha=0.2)

plt.show()