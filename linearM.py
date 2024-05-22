############################################
#
#  Data Science:  Linear Model Analysis
#
#  Written By : BARA AHMAD MOHAMMED
#
#############################################

# TODO : IMPORT ALL NEEDED LIBRARIES
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Ensure the 'images' directory exists
os.makedirs('images', exist_ok=True)

# Load the data from the CSV file
data = pd.read_csv('./data/hw3.data1.csv')

# Extract features (X) and labels (y) from the dataset
X = data.drop(columns=['label']).values
y_actual = data['label'].values

# Define the linear model coefficients and intercept
coefficients = np.array([24, -15, -38, -7, -41, 35, 0, -2, 19, 33, -3, 7, 3, -47, 26, 10, 40, -1, 3, 0])
intercept = -6  #{ [6] , [10] , [-10]}

# Define the linear model prediction function with adjustable threshold
def predict_linear_model_with_threshold(X, threshold=0):
    y_pred = np.dot(X, coefficients) + intercept
    return np.where(y_pred > threshold, 1, -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_actual, test_size=0.2, random_state=42)

# Define the cost parameters
cost_false_negative = 1000
cost_false_positive = 100

# Find the best threshold to minimize economic loss
best_threshold = 0
best_economic_loss = float('inf')

for threshold in np.arange(-10, 10, 0.5):
    y_pred = predict_linear_model_with_threshold(X_test, threshold)
    conf_matrix = confusion_matrix(y_test, y_pred)
    false_negatives = conf_matrix[1, 0]
    false_positives = conf_matrix[0, 1]
    economic_loss = (cost_false_negative * false_negatives) + (cost_false_positive * false_positives)
    if economic_loss < best_economic_loss:
        best_economic_loss = economic_loss
        best_threshold = threshold

# Predict using the best threshold
y_pred = predict_linear_model_with_threshold(X_test, best_threshold)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate false negatives and false positives from confusion matrix
false_negatives = conf_matrix[1, 0]
false_positives = conf_matrix[0, 1]

# Calculate economic gain
economic_loss = (cost_false_negative * false_negatives) + (cost_false_positive * false_positives)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Economic Loss:", economic_loss)
print("Best Threshold:", best_threshold)

# Plotting the predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.grid(True)
plt.savefig('images/actual_vs_predicted.png')
plt.show()

# Plotting the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted -1', 'Predicted 1'], yticklabels=['Actual -1', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('images/confusion_matrix.png')
plt.show()
