import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri('http://127.0.0.1:5000')

wine = load_wine()

X = wine.data
y = wine.target

trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 5
n_estimators = 5

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(trainx, trainy)
    
    rf_pred = rf.predict(testx)
    
    print(accuracy_score(rf_pred, testy))
    
    mlflow.log_metric('accuracy', accuracy_score(rf_pred, testy))
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    
    mlflow.log_artifact(__file__)