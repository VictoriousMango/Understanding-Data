from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.ensemble import RandomForestRegressor  # Change this import
from sklearn.metrics import mean_squared_error  # Import for regression evaluation

def randomForest(featuresName, targetVar, test_size=0.2):
    randomState = 0
    df = pd.read_csv("./Data/Data.csv")
    columnNames = [[columnName, 0] for columnName in df.columns if columnName in featuresName]
    X = df[featuresName]
    y = df[targetVar]
    
    # Test/Train Split of the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=randomState)
    
    # Model
    model = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=randomState)  # Use Regressor
    
    # Model Fit
    model.fit(X_train, y_train)
    for i in range(len(model.feature_importances_)):
        columnNames[i][1] += model.feature_importances_[i]
    # Predict
    y_pred = model.predict(X_test)  # Get predictions

    # Calculate Mean Squared Error (or any other regression metric)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # # AUC ROC Curve
    # threshold = 0.5 # Define Threshold
    # y_pred_binary = (y_pred > threshold).astype(int)
    # fpr, tpr, _ = roc_curve(y_test, y_pred_binary)
    # roc_auc = auc(fpr, tpr)

    # Plotting Predicted vs Actual
    plt.figure(figsize=(12, 6))
    plt.title("Random Forest Classifier")

    # Scatter plot of predicted vs actual values
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    
    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')

    # # AUC ROC plot
    # plt.subplot(1, 3, 3)
    # plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic for Random Forest Regressor')
    # plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('./static/images/ModelEvaluation.png')
    
    return (columnNames, featuresName, targetVar)

if __name__ == "__main__":
    randomForest(targetVar="Air Quality", featuresName=["Humidity", "PM2.5", "Proximity_to_Industrial_Areas", "Population_Density"], test_size=0.3)