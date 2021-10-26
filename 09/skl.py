"""
Machine Learning Modelling Tutorial with Python and Scikit-Learn
"""
from os.path import join, dirname
from pickle import dump as pdump, load as pload
from joblib import dump as jdump, load as jload
from pandas import DataFrame, Series, read_csv, get_dummies, crosstab
from numpy import mean, zeros, full, arange
from numpy.random import seed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
    roc_curve,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
    cross_val_score,
)
from matplotlib.pyplot import (
    plot,
    figure,
    xlabel,
    ylabel,
    title,
    legend,
    show,
    subplots,
)

file = lambda x: join(dirname(__file__), x)


def plot_roc_curve(fpr, tpr):
    """
    Plots a ROC curve given the false positve rate (fpr) and
    true postive rate (tpr) of a classifier.
    """
    figure()
    plot(fpr, tpr, color="orange", label="ROC")
    plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Guessing")
    xlabel("False Positive Rate")
    ylabel("True Positive Rate")
    title("Receiver Operating Characteristic (ROC) Curve")
    legend()


def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1s = f1_score(y_true, y_preds)
    metric_dict = {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1s, 2),
    }
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1s:.2f}")

    return metric_dict


def wrapper():
    """
    wrapper function
    """
    heart_disease = DataFrame(read_csv(file("heart-disease.csv")))
    print(heart_disease.head())
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    print(xda.head())
    print(yda.head(), yda.value_counts())
    x_train, x_test, y_train, y_test = train_test_split(xda, yda)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    clf = RandomForestClassifier()
    print(clf.get_params())
    clf.fit(x_train, y_train)
    print(x_test.head())
    y_preds = clf.predict(x_test)
    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))
    print(classification_report(y_test, y_preds))
    conf_mat = confusion_matrix(y_test, y_preds)
    print(conf_mat)
    seed(42)
    for i in range(10, 100, 10):
        print(f"Trying model with {i} estimators...")
        model = RandomForestClassifier(n_estimators=i).fit(x_train, y_train)
        print(f"Model accuracy on test set: {model.score(x_test, y_test) * 100}%")
        print("")
    seed(41)
    for i in range(10, 100, 10):
        print(f"Trying model with {i} estimators...")
        model = RandomForestClassifier(n_estimators=i).fit(x_train, y_train)
        print(f"Model accuracy on test set: {model.score(x_test, y_test) * 100}%")
        print(
            f"Cross-validation score: {mean(cross_val_score(model, xda, yda, cv=5)) * 100}%"
        )
        print("")
    seed(40)
    param_grid = {"n_estimators": list(range(10, 100, 10))}
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid.fit(xda, yda)
    print(grid.best_params_)
    clf = grid.best_estimator_
    print(clf)
    clf = clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))

    pickle_func(model, heart_disease, x_test, y_test)


def pickle_func(model, heart_disease, x_test, y_test):
    """
    Save an existing model to file
    Load a saved model and make a prediction
    """
    with open(file("random_forest_model_1.pkl"), "wb") as rfm:
        pdump(model, rfm)
    with open(file("random_forest_model_1.pkl"), "rb") as rfm:
        print(pload(rfm).score(x_test, y_test))
    print(heart_disease.head())
    xda = heart_disease.drop("target", axis=1)
    print(xda)
    yda = heart_disease["target"]
    print(yda)
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print(xda.shape[0] * 0.8)
    car_sales = DataFrame(read_csv(file("car-sales-extended.csv")))
    print(car_sales)
    print(car_sales.dtypes)
    xda = car_sales.drop("Price", axis=1)
    yda = car_sales["Price"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    categorical_features = ["Make", "Colour", "Doors"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer(
        [("one_hot", one_hot, categorical_features)], remainder="passthrough"
    )
    transformed_x = transformer.fit_transform(xda)
    print(transformed_x)
    print(transformed_x[0])
    print(xda.iloc[0])
    print(car_sales.head())
    dummies = get_dummies(car_sales[["Make", "Colour", "Doors"]])
    print(dummies)
    car_sales["Doors"] = car_sales["Doors"].astype(object)
    dummies = get_dummies(car_sales[["Make", "Colour", "Doors"]])
    print(dummies)
    print(xda["Make"].value_counts())

    categories(transformed_x, yda, model)


def categories(transformed_x, yda, model):
    """
    Turn the categories (Make and Colour) into numbers
    """
    seed(39)
    x_train, x_test, y_train, y_test = train_test_split(
        transformed_x, yda, test_size=0.2
    )
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    car_sales_missing = DataFrame(read_csv(file("car-sales-extended-missing-data.csv")))
    print(car_sales_missing)
    print(car_sales_missing.isna().sum())
    categorical_features = ["Make", "Colour", "Doors"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer(
        [("one_hot", one_hot, categorical_features)], remainder="passthrough"
    )
    transformed_x = transformer.fit_transform(car_sales_missing)
    print(transformed_x)
    print(car_sales_missing.isna().sum())
    car_sales_missing["Make"].fillna("missing", inplace=True)
    car_sales_missing["Colour"].fillna("missing", inplace=True)
    car_sales_missing["Odometer (KM)"].fillna(
        car_sales_missing["Odometer (KM)"].mean(), inplace=True
    )
    car_sales_missing["Doors"].fillna(4, inplace=True)
    print(car_sales_missing.isna().sum())
    car_sales_missing.dropna(inplace=True)
    print(car_sales_missing.isna().sum())
    print(len(car_sales_missing))
    categorical_features = ["Make", "Colour", "Doors"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer(
        [("one_hot", one_hot, categorical_features)], remainder="passthrough"
    )
    transformed_x = transformer.fit_transform(car_sales_missing)
    print(transformed_x)
    print(transformed_x[0])
    print(car_sales_missing.isna().sum())
    car_sales_missing = DataFrame(read_csv(file("car-sales-extended-missing-data.csv")))
    print(car_sales_missing.isna().sum())
    car_sales_missing.dropna(subset=["Price"], inplace=True)
    print(car_sales_missing.isna().sum())
    xda = car_sales_missing.drop("Price", axis=1)
    yda = car_sales_missing["Price"]

    imputers(xda, yda, car_sales_missing)


def imputers(xda, yda, car_sales_missing):
    """
    Fill categorical values with 'missing' & numerical with mean
    """
    seed(38)
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    door_imputer = SimpleImputer(strategy="constant", fill_value=4)
    num_imputer = SimpleImputer(strategy="mean")
    imputer = ColumnTransformer(
        [
            ("cat_imputer", cat_imputer, ["Make", "Colour"]),
            ("door_imputer", door_imputer, ["Doors"]),
            ("num_imputer", num_imputer, ["Odometer (KM)"]),
        ]
    )
    filled_x_train = imputer.fit_transform(x_train)
    filled_x_test = imputer.transform(x_test)
    print(filled_x_train)

    ridges(filled_x_train, filled_x_test, car_sales_missing, y_train, y_test)


def ridges(filled_x_train, filled_x_test, car_sales_missing, y_train, y_test):
    """
    Import the Ridge model class from the linear_model module
    """
    car_sales_filled_train = DataFrame(
        filled_x_train, columns=["Make", "Colour", "Doors", "Odometer (KM)"]
    )
    car_sales_filled_test = DataFrame(
        filled_x_test, columns=["Make", "Colour", "Doors", "Odometer (KM)"]
    )
    print(car_sales_filled_train.isna().sum())
    print(car_sales_missing.isna().sum())
    categorical_features = ["Make", "Colour", "Doors"]
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer(
        [("one_hot", one_hot, categorical_features)], remainder="passthrough"
    )
    transformed_x_train = transformer.fit_transform(car_sales_filled_train)
    transformed_x_test = transformer.transform(car_sales_filled_test)
    print(transformed_x_train.toarray())
    seed(37)
    model = RandomForestRegressor()
    model.fit(transformed_x_train, y_train)
    print(model.score(transformed_x_test, y_test))
    housing = fetch_california_housing()
    print(housing)
    housing_df = DataFrame(housing["data"], columns=housing["feature_names"])
    housing_df["target"] = Series(housing["target"])
    print(housing_df.head())
    print(len(housing_df))

    linearsvc(housing_df)


def linearsvc(housing_df):
    """
    Import LinearSVC from the svm module
    """
    seed(36)
    xda = housing_df.drop("target", axis=1)
    yda = housing_df["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    model = Ridge()
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    seed(35)
    xda = housing_df.drop("target", axis=1)
    yda = housing_df["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    heart_disease = DataFrame(read_csv(file("heart-disease.csv")))
    print(heart_disease.head())
    print(len(heart_disease))
    seed(34)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    clf = LinearSVC(max_iter=1000)
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    seed(33)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    seed(32)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    print(xda.head())
    print(yda.head())
    print(clf.predict(x_test))
    y_preds = clf.predict(x_test)
    mean(y_preds == y_test)
    print(accuracy_score(y_test, y_preds))
    print(clf.predict_proba(x_test[:5]))
    print(clf.predict(x_test[:5]))
    print(clf.predict_proba(x_test[:1]))
    print(clf.predict(x_test[:1]))

    randomforest(housing_df, heart_disease)


def randomforest(housing_df, heart_disease):
    """
    Import the RandomForestRegressor model class from the ensemble module
    """
    seed(31)
    xda = housing_df.drop("target", axis=1)
    yda = housing_df["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    y_preds = model.predict(x_test)
    print(mean_absolute_error(y_test, y_preds))
    seed(30)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    seed(29)
    xda = housing_df.drop("target", axis=1)
    yda = housing_df["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    seed(28)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    print(cross_val_score(clf, xda, yda))
    print(cross_val_score(clf, xda, yda, cv=5))
    seed(27)
    clf_single_score = clf.score(x_test, y_test)
    clf_cross_val_score = mean(cross_val_score(clf, xda, yda, cv=5))
    print(clf_single_score, clf_cross_val_score)
    print(cross_val_score(clf, xda, yda, cv=5, scoring=None))

    crossval(heart_disease, housing_df)


def crossval(heart_disease, housing_df):
    """
    Import cross_val_score from the model_selection module
    """
    seed(26)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    clf.score(x_test, y_test)
    print(f"Heart Disease Classifier Accuracy: {clf.score(x_test, y_test) * 100:.2f}%")
    y_probs = clf.predict_proba(x_test)
    y_probs = y_probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    print(fpr)
    plot_roc_curve(fpr, tpr)
    print(roc_auc_score(y_test, y_probs))
    fpr, tpr, _ = roc_curve(y_test, y_test)
    plot_roc_curve(fpr, tpr)
    print(roc_auc_score(y_test, y_test))
    y_preds = clf.predict(x_test)
    print(confusion_matrix(y_test, y_preds))
    print(
        crosstab(
            y_test, y_preds, rownames=["Actual Label"], colnames=["Predicted Label"]
        )
    )
    figure()
    ConfusionMatrixDisplay.from_estimator(estimator=clf, X=xda, y=yda)
    ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_preds)
    print(classification_report(y_test, y_preds))

    randomfr(housing_df, heart_disease)


def randomfr(housing_df, heart_disease):
    """
    Import the RandomForestRegressor model class from the ensemble module
    """
    disease_true = zeros(10000)
    disease_true[0] = 1
    disease_preds = zeros(10000)
    print(
        DataFrame(
            classification_report(
                disease_true, disease_preds, output_dict=True, zero_division=0
            )
        )
    )
    seed(25)
    xda = housing_df.drop("target", axis=1)
    yda = housing_df["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    y_test_mean = full(len(y_test), y_test.mean())
    print(r2_score(y_test, y_test_mean))
    print(r2_score(y_test, y_test))
    y_preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_preds)
    print(mae)
    daf = DataFrame(data={"actual values": y_test, "predictions": y_preds})
    print(daf)

    fscore(daf, y_test, y_preds, housing_df, heart_disease)


def fscore(daf, y_test, y_preds, housing_df, heart_disease):
    """
    f1 for fscore function
    """
    _, axi = subplots()
    xda = arange(0, len(daf), 1)
    axi.scatter(xda, daf["actual values"], c="b", label="Acutual Values")
    axi.scatter(xda, daf["predictions"], c="r", label="Predictions")
    axi.legend(loc=(1, 0.5))
    mse = mean_squared_error(y_test, y_preds)
    print(mse)
    seed(24)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    clf = RandomForestClassifier(n_estimators=100)
    seed(23)
    cv_acc = cross_val_score(clf, xda, yda, cv=5)
    print(cv_acc)
    print(f"The cross-validated accuracy is: {mean(cv_acc)*100:.2f}%")
    seed(22)
    cv_acc = cross_val_score(clf, xda, yda, cv=5, scoring="accuracy")
    print(f"The cross-validated accuracy is: {mean(cv_acc)*100:.2f}%")
    seed(21)
    cv_precision = cross_val_score(clf, xda, yda, cv=5, scoring="precision")
    print(f"The cross-validated precision is: {mean(cv_precision):.2f}")
    seed(20)
    cv_recall = cross_val_score(clf, xda, yda, cv=5, scoring="recall")
    print(f"The cross-validated recall is: {mean(cv_recall):.2f}")
    seed(19)
    cv_f1 = cross_val_score(clf, xda, yda, cv=5, scoring="f1")
    print(f"The cross-validated F1 score is: {mean(cv_f1):.2f}")

    heart_func(heart_disease, housing_df)


def heart_func(heart_disease, housing_df):
    """
    heart disease function
    """
    seed(18)
    xda = housing_df.drop("target", axis=1)
    yda = housing_df["target"]
    model = RandomForestRegressor(n_estimators=100)
    seed(17)
    cv_r2 = cross_val_score(model, xda, yda, cv=5, scoring="r2")
    print(f"The cross-validated R^2 score is: {mean(cv_r2):.2f}")
    seed(16)
    cv_mae = cross_val_score(model, xda, yda, cv=5, scoring="neg_mean_absolute_error")
    print(f"The cross-validated MAE score is: {mean(cv_mae):.2f}")
    seed(15)
    cv_mse = cross_val_score(model, xda, yda, cv=5, scoring="neg_mean_squared_error")
    print(f"The cross-validated MSE score is: {mean(cv_mse):.2f}")
    seed(14)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)
    print("Classifier metrics on the test set:")
    print(f"Accuracy: {accuracy_score(y_test, y_preds) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_preds):.2f}")
    print(f"Recall: {recall_score(y_test, y_preds):.2f}")
    print(f"F1: {f1_score(y_test, y_preds):.2f}")
    seed(13)
    xda = housing_df.drop("target", axis=1)
    yda = housing_df["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(x_train, y_train)
    y_preds = model.predict(x_test)
    print("Regression model metrics on the test set:")
    print(f"R^2: {r2_score(y_test, y_preds):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_preds):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_preds):.2f}")
    clf = RandomForestClassifier()
    print(clf.get_params())
    seed(12)

    heart_disease_func(heart_disease)


def heart_disease_func(heart_disease):
    """
    heart disease function
    """
    heart_disease = heart_disease.sample(frac=1)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    train_split = round(0.7 * len(heart_disease))
    valid_split = round(train_split + 0.15 * len(heart_disease))
    x_train, y_train = xda[:train_split], yda[:train_split]
    x_valid, y_valid = xda[train_split:valid_split], yda[train_split:valid_split]
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_valid)
    baseline_metrics = evaluate_preds(y_valid, y_preds)
    print(baseline_metrics)
    seed(11)
    clf_2 = RandomForestClassifier(n_estimators=100)
    clf_2.fit(x_train, y_train)
    y_preds_2 = clf_2.predict(x_valid)
    clf_2_metrics = evaluate_preds(y_valid, y_preds_2)

    heartdise_func(heart_disease, baseline_metrics, clf_2_metrics)


def heartdise_func(heart_disease, baseline_metrics, clf_2_metrics):
    """
    heart disease function
    """
    grid = {
        "n_estimators": [10, 100, 200, 500, 1000, 1200],
        "max_depth": [None, 5, 10, 20, 30],
        "max_features": ["auto", "sqrt"],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4],
    }
    seed(10)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    clf = RandomForestClassifier(n_jobs=1)
    rs_clf = RandomizedSearchCV(
        estimator=clf, param_distributions=grid, n_iter=20, cv=5, verbose=2
    )
    rs_clf.fit(x_train, y_train)
    seed(9)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    clf = RandomForestClassifier(n_jobs=1)
    rs_clf = RandomizedSearchCV(
        estimator=clf, param_distributions=grid, n_iter=20, cv=5, verbose=2
    )
    rs_clf.fit(x_train, y_train)
    print(rs_clf.best_params_)
    rs_y_preds = rs_clf.predict(x_test)
    rs_metrics = evaluate_preds(y_test, rs_y_preds)
    print(grid)

    heart_disease_last(heart_disease, baseline_metrics, clf_2_metrics, rs_metrics)


def heart_disease_last(heart_disease, baseline_metrics, clf_2_metrics, rs_metrics):
    """
    heart disease function
    """
    grid_2 = {
        "n_estimators": [1200, 1500, 2000],
        "max_depth": [None, 5, 10],
        "max_features": ["auto", "sqrt"],
        "min_samples_split": [4, 6],
        "min_samples_leaf": [1, 2],
    }
    seed(8)
    xda = heart_disease.drop("target", axis=1)
    yda = heart_disease["target"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    clf = RandomForestClassifier(n_jobs=1)
    gs_clf = GridSearchCV(estimator=clf, param_grid=grid_2, cv=5, verbose=2)
    gs_clf.fit(x_train, y_train)
    print(gs_clf.best_params_)
    gs_y_preds = gs_clf.predict(x_test)
    gs_metrics = evaluate_preds(y_test, gs_y_preds)
    DataFrame(
        {
            "baseline": baseline_metrics,
            "clf_2": clf_2_metrics,
            "random search": rs_metrics,
            "grid search": gs_metrics,
        }
    ).plot.bar(figsize=(10, 8))

    open_close(gs_clf, x_test, y_test)


def open_close(gs_clf, x_test, y_test):
    """
    car sales extended function
    """
    with open(file("gs_random_forest_model_1.pkl"), "wb") as grf:
        pdump(gs_clf, grf)
    with open(file("gs_random_forest_model_1.pkl"), "rb") as grf:
        loaded_pickle_model = pload(grf)
    pickle_y_preds = loaded_pickle_model.predict(x_test)
    print(evaluate_preds(y_test, pickle_y_preds))
    jdump(gs_clf, filename="gs_random_forest_model_1.joblib")
    loaded_joblib_model = jload(filename="gs_random_forest_model_1.joblib")
    joblib_y_preds = loaded_joblib_model.predict(x_test)
    evaluate_preds(y_test, joblib_y_preds)
    data = DataFrame(read_csv(file("car-sales-extended-missing-data.csv")))
    print(data)
    print(data.dtypes)
    print(data.isna().sum())

    car_sales_miss()


def car_sales_miss():
    """
    car sales missing function
    """
    seed(7)
    data = DataFrame(read_csv(file("car-sales-extended-missing-data.csv")))
    data.dropna(subset=["Price"], inplace=True)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    door_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=4))]
    )
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, ["Make", "Colour"]),
            ("door", door_transformer, ["Doors"]),
            ("num", numeric_transformer, ["Odometer (KM)"]),
        ]
    )
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor())]
    )
    xda = data.drop("Price", axis=1)
    yda = data["Price"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    pipe_grid = {
        "preprocessor__num__imputer__strategy": ["mean", "median"],
        "model__n_estimators": [100, 1000],
        "model__max_depth": [None, 5],
        "model__max_features": ["auto", "sqrt"],
        "model__min_samples_split": [2, 4],
    }
    gs_model = GridSearchCV(model, pipe_grid, cv=5, verbose=2)
    gs_model.fit(x_train, y_train)
    print(gs_model.score(x_test, y_test))


if __name__ == "__main__":
    wrapper()
    show()
