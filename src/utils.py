import os
import sys
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name in models:

            model = models[model_name]
            para = param[model_name]

            if len(para) != 0:
                gs = GridSearchCV(
                    model,
                    para,
                    cv=3,
                    n_jobs=-1,
                    verbose=0
                )

                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)