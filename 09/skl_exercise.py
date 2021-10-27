"""
Scikit-Learn Practice
"""
from os.path import join, dirname
from pandas import DataFrame, read_csv
from joblib import dump, load
from numpy import logspace, mean
from numpy.random import seed
from matplotlib.pyplot import show, subplots, xlabel, ylabel
from seaborn import heatmap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    plot_roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
    cross_val_score,
)

file = lambda x: join(dirname(__file__), x)


def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    _, axi = subplots(figsize=(3, 3))
    axi = heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
    xlabel("True label")
    ylabel("Predicted label")
    bottom, top = axi.get_ylim()
    axi.set_ylim(bottom + 0.5, top - 0.5)


def wrapper():
    """
    wrapper function
    """
    heart_disease = DataFrame(read_csv(file("heart-disease.csv")))
    print(heart_disease.head())
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    print(clf.predict(x_test))
    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))
    example_dict = {"RandomForestClassifier": RandomForestClassifier()}
    models = {
        "LinearSVC": LinearSVC(),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(),
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(),
    }
    results = {}
    example_results = {}
    for model_name, model in example_dict.items():
        model.fit(x_train, y_train)
        example_results[model_name] = model.score(x_test, y_test)
    print(example_results)
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        results[model_name] = model.score(x_test, y_test)
    print(results)
    seed(42)
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        results[model_name] = model.score(x_test, y_test)
    print(results)
    results_df = DataFrame(results.values(), results.keys(), columns=["Accuracy"])
    results_df.plot.bar()

    rscv()


def rscv():
    """
    Setup an instance of RandomizedSearchCV with a LogisticRegression() estimator,
    our log_reg_grid as the param_distributions, a cv of 5 and n_iter of 5.
    """
    heart_disease = DataFrame(read_csv(file("heart-disease.csv")))
    print(heart_disease.head())
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda)
    log_reg_grid = {"C": logspace(-4, 4, 20), "solver": ["liblinear"]}
    seed(41)
    rs_log_reg = RandomizedSearchCV(
        estimator=LogisticRegression(),
        param_distributions=log_reg_grid,
        cv=5,
        n_iter=5,
        verbose=True,
    )
    rs_log_reg.fit(x_train, y_train)
    print(rs_log_reg.best_params_)
    print(rs_log_reg.score(x_test, y_test))
    clf = LogisticRegression(solver="liblinear", C=0.23357214690901212)
    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)
    print(confusion_matrix(y_test, y_preds))
    plot_conf_mat(y_test, y_preds)
    print(classification_report(y_test, y_preds))
    print(precision_score(y_test, y_preds))
    print(recall_score(y_test, y_preds))
    print(f1_score(y_test, y_preds))
    plot_roc_curve(clf, x_test, y_test)
    print(cross_val_score(clf, xda, yda, scoring="accuracy", cv=5))
    cross_val_acc = mean(cross_val_score(clf, xda, yda, scoring="accuracy", cv=5))
    print(cross_val_acc)
    cross_val_precision = mean(
        cross_val_score(clf, xda, yda, scoring="precision", cv=5)
    )
    print(cross_val_precision)
    cross_val_recall = mean(cross_val_score(clf, xda, yda, scoring="recall", cv=5))
    print(cross_val_recall)
    cross_val_f1 = mean(cross_val_score(clf, xda, yda, scoring="f1", cv=5))
    print(cross_val_f1)

    carsales(clf, x_test, y_test)


def carsales(clf, x_test, y_test):
    """
    Scikit-Learn Regression Practice
    """
    dump(clf, file("trained-classifier.joblib"))
    loaded_clf = load(file("trained-classifier.joblib"))
    loaded_clf.score(x_test, y_test)
    car_sales = DataFrame(read_csv(file("car-sales-extended-missing-data.csv")))
    print(car_sales.head())
    print(car_sales.info())
    print(car_sales.isna().sum())
    print(car_sales.dtypes)
    car_sales.dropna(subset=["Price"], inplace=True)
    categorical_features = ["Make", "Colour"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    door_feature = ["Doors"]
    door_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=4))]
    )
    numeric_features = ["Odometer (KM)"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("door", door_transformer, door_feature),
            ("num", numeric_transformer, numeric_features),
        ]
    )
    regression_models = {
        "Ridge": Ridge(),
        "SVR_linear": SVR(kernel="linear"),
        "SVR_rbf": SVR(kernel="rbf"),
        "RandomForestRegressor": RandomForestRegressor(),
    }
    regression_results = {}

    car_model(car_sales, regression_models, regression_results, preprocessor)


def car_model(car_sales, regression_models, regression_results, preprocessor):
    """
    Loop through the items in the regression_models dictionary
    """
    car_sales_x = car_sales.drop("Price", axis=1)
    car_sales_y = car_sales["Price"]
    car_x_train, car_x_test, car_y_train, car_y_test = train_test_split(
        car_sales_x, car_sales_y, test_size=0.2, random_state=42
    )
    print(car_x_train.shape, car_x_test.shape, car_y_train.shape, car_y_test.shape)
    for model_name, model in regression_models.items():
        model_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )
        print(f"Fitting {model_name}...")
        model_pipeline.fit(car_x_train, car_y_train)
        print(f"Scoring {model_name}...")
        regression_results[model_name] = model_pipeline.score(car_x_test, car_y_test)
    print(regression_results)
    ridge_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", Ridge())]
    )
    ridge_pipeline.fit(car_x_train, car_y_train)
    car_y_preds = ridge_pipeline.predict(car_x_test)
    print(car_y_preds[:50])
    print(mean_squared_error(car_y_test, car_y_preds))
    print(mean_absolute_error(car_y_test, car_y_preds))
    print(r2_score(car_y_test, car_y_preds))


if __name__ == "__main__":
    wrapper()
    show()
