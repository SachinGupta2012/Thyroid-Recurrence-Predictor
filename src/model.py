from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

def train_model(X, y, model_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        "XGBoost": (XGBClassifier(eval_metric='logloss'), {
            'n_estimators': [100], 'max_depth': [3], 'learning_rate': [0.1]
        }),
        "LogisticRegression": (LogisticRegression(), {'C': [10]}),
        "RandomForest": (RandomForestClassifier(), {'n_estimators': [200]}),
        "SVM": (SVC(), {'C': [1], 'kernel': ['linear']}),
        "KNN": (KNeighborsClassifier(), {'n_neighbors': [3]})
    }

    best_score = 0
    best_model = None

    for name, (model, params) in models.items():
        clf = GridSearchCV(model, params, cv=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = f1_score(y_test, y_pred)
        print(f"{name} | Accuracy: {accuracy_score(y_test, y_pred):.4f} | F1 Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = clf.best_estimator_

    joblib.dump(best_model, model_path)
    print("Model saved to", model_path)