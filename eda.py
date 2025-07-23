import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(filepath):
    df = pd.read_csv(filepath)
    print(df.describe())
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('outputs/correlation_heatmap.png')
    plt.clf()
    df.hist(figsize=(10, 8))
    plt.tight_layout()
    plt.savefig('outputs/histograms.png')
    plt.clf()
