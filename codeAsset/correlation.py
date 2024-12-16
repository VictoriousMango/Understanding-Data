import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def correlation():
    df = pd.read_csv("./Data/Data.csv")
    corr_matrix = df.corr()
    corr_matrix.to_csv("./Data/CorrelationData.csv")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_matrix, mask=np.triu(np.ones_like(corr_matrix, dtype=bool)), cmap='coolwarm', square=True)
    # Save the plot as an image file
    plt.savefig('./static/images/correlation_plot.png')
    print(type(corr_matrix))
    table = corr_matrix.to_dict(orient="records")
    return (table, df.columns)