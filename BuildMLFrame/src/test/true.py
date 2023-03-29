import mlfoundry
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_iris(as_frame=True, return_X_y=True)
X = X.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True))])
pipe.fit(X_train, y_train)

client = mlfoundry.get_client()
run = client.create_run(project_name="iris-clf", run_name="iris-svc")

# Optionally, we can log hyperparameters using run.log_params(...)
# Optionally, we can log metrics using run.log_metrics(...)

model_version = run.log_model(name="iris-svc", model=pipe, framework="sklearn", step=0)
print("Model Logged as:", model_version.fqn)