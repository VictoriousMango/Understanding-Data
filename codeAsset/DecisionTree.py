from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, mean_squared_error, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
randomForest:
- parameters: 
    featureName: Name of Features upon which the model is supposed to be trained, 
    targetVar  : Name of the target variable or the dependent variable, which is to be predicted using featureName, 
    test_size  : Size of the test dataset 
- preRequiste: existing Data.csv in Data folder
- test-Train Splitter : 
- input: X_train, y_train, X_test, y_test
- output: columnNames, featuresName, targetVar

"""

def decisionTree(featuresName, targetVar, test_size=0.2):
    randomState = 0
    df = pd.read_csv("./Data/Data.csv")
    columnNames = [[columnName, 0] for columnName in df.columns if columnName in featuresName]
    X = df[featuresName]
    y = df[targetVar]
    
    # Test/Train Split of the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=randomState)
    
    # Model
    model = DecisionTreeRegressor(max_depth=2, random_state=randomState)

    # Model Fit
    model.fit(X_train, y_train)
    for i in range(len(model.feature_importances_)):
        columnNames[i][1] += model.feature_importances_[i]
    

    # Predict
    y_pred = model.predict(X_test)  # Get predictions

    # Calculate Mean Squared Error (or any other regression metric)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot the ROC curve
    os.remove("./static/images/ModelEvaluation.png")
   # Plotting Predicted vs Actual
    plt.figure(figsize=(12, 6))
    plt.title("Decision Tree Regressor")

    
    # Scatter plot of predicted vs actual values
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    
    # Residual plot
    plt.subplot(2, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')

    # Model Visualization
    plt.subplot(2, 2, 3)
    tree.plot_tree(model)

    plt.tight_layout()
    # plt.show()
    plt.savefig('./static/images/ModelEvaluation.png')
    
    return (columnNames, featuresName, targetVar)

if __name__ == "__main__":
    decisionTree(targetVar="Air Quality", featuresName=["Humidity", "PM2.5", "Proximity_to_Industrial_Areas", "Population_Density"], test_size=0.3)