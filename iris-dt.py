import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import dagshub


# ------------------ Init DagsHub + MLflow ------------------ #
dagshub.init(repo_owner='Vaibha3246', repo_name='mlflow-dagshub-demo', mlflow=True)

# ✅ Correct tracking URI for DagsHub
mlflow.set_tracking_uri("https://dagshub.com/Vaibha3246/mlflow-dagshub-demo.mlflow")

# Set experiment
experiment_name = "iris-dt"
mlflow.set_experiment(experiment_name)


# ------------------ Load Data ------------------ #
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ------------------ Model Params ------------------ #
max_depth = 4



# ------------------ Train + Log ------------------ #
with mlflow.start_run() as run:
    run_id = run.info.run_id

    # Train model
    dt= DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric('accuracy', accuracy)

    # Params
    mlflow.log_param('max_depth', max_depth)
    

    # ------------------ Confusion Matrix ------------------ #
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")

    # Log confusion matrix as artifact
    mlflow.log_artifact("confusion_matrix.png")

    # ------------------ Save & Log Model ------------------ #
    model_path = "descion_tree_model.pkl"
    joblib.dump(dt, model_path)                  # save locally
    mlflow.log_artifact(model_path)              # upload to DagsHub artifacts

    # ------------------ Tags ------------------ #
    mlflow.set_tag('author', 'mahesh')
    mlflow.set_tag('model', 'descion tree classifier')

    print(f'✅ Run {run_id} completed with accuracy: {accuracy:.4f}')