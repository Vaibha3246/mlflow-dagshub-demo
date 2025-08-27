import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
from mlflow.exceptions import MlflowException

# Initialize DagsHub
dagshub.init(repo_owner='Vaibha3246', repo_name='mlflow-dagshub-demo', mlflow=True)

# Correct tracking URI (must point to DagsHub, not GitHub!)
mlflow.set_tracking_uri("https://dagshub.com/Vaibha3246/mlflow-dagshub-demo.mlflow")

# Set or create experiment
experiment_name = "iris-rf"
try:
    mlflow.set_experiment(experiment_name)
except MlflowException:
    mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model params
max_depth = 1
n_estimators = 100

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # Log model
    mlflow.sklearn.log_model(rf, "random_forest")

    # Tags
    mlflow.set_tag('author','rahul')
    mlflow.set_tag('model','random forest')

    print('accuracy', accuracy)
