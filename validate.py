import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report



# Replace the dataset with your validation dataset
merged_train=pd.read_csv("merged_train.csv") #input the transformed train
merged_test=pd.read_csv("merged_test.csv") #input the transformed test

X_train=merged_train.drop(['Unnamed: 0','target'],axis=1)
y_train=merged_train['target']
X_test=merged_test.drop(['Unnamed: 0','target'],axis=1)
y_test=merged_test['target']



lightgbm_params = {
    'n_estimators': 100,              # Number of boosting iterations
    'learning_rate': 0.1,             # Step size
    'num_leaves': 31,                 # Maximum number of leaves in one tree
    'max_depth': -1,                  # Maximum depth of the tree, -1 means no limit
    'scale_pos_weight': 1,            # Balancing of positive and negative weights
}

models = {
    "LightGBM": LGBMClassifier(**lightgbm_params)   
}
# Function to evaluate model performance
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("\n" + "-"*50 + "\n")
    joblib.dump(model,"lgbm.pkl")
    
    
    return accuracy, conf_matrix, precision, recall, f1

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    
    results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)


