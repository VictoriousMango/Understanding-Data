from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def multicolinearity(columns):
    df = pd.read_csv("./Data/Data.csv")

    # VIF dataframe
    vif_data = pd.DataFrame()
    if columns:
        df = df.drop(columns, axis=1)
    vif_data["feature"] = df.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                            for i in range(len(df.columns))]
    vif = [[vif_data["feature"][i], vif_data["VIF"][i]] for i in range(len(df.columns))]
    return vif