# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import Libraries
Import pandas, scikit-learn, seaborn, and matplotlib for data handling, modeling, and visualization.
2. Load Dataset
Load the dataset from the provided URL and explore its structure.
3. Feature Selection
Define features (X) by excluding irrelevant columns like id and set diagnosis as the target (y).
4. Split Dataset
Split the data into training and testing sets (70-30 ratio).
5. Train Model
Train a Decision Tree Classifier on the training data.
6. Evaluate Model
Predict using the test set and calculate accuracy, generate a classification report, and confusion matrix.
7. Visualize Results
Plot a heatmap of the confusion matrix to assess prediction performance.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('tumor.csv')

print(data.head())
print(data.columns)

X = data.drop(columns=['Class'])  
y = data['Class'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nName: Arunjuthan.M.A")
print("Reg No: 212225230020")
print("\nAccuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="913" height="585" alt="image" src="https://github.com/user-attachments/assets/4c999553-e6d2-4733-8dd8-78df5e54f193" />

<img width="747" height="513" alt="image" src="https://github.com/user-attachments/assets/1c2d810c-67fb-4530-8efe-324c5c5e50a3" />

## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
