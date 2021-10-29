"""
Predicting Heart Disease using Machine Learning
"""
from os.path import join, dirname
from types import SimpleNamespace
from pandas import DataFrame, read_csv, crosstab
from seaborn import heatmap, set as sset
from numpy import arange, logspace, mean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    train_test_split,
    cross_val_score,
)
from matplotlib.pyplot import (
    plot,
    subplots,
    show,
    title,
    xlabel,
    ylabel,
    legend,
    xticks,
    scatter,
    figure,
)

file = lambda x: join(dirname(__file__), x)


def fit_and_score(models, x_train, x_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    x_train : training data
    x_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    model_scores = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        model_scores[name] = model.score(x_test, y_test)
    return model_scores


def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    subplots(figsize=(3, 3))
    heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
    xlabel("true label")
    ylabel("predicted label")


def wrapper():
    """
    wrapper function
    """
    daf = DataFrame(read_csv(file("heart-disease.csv")))
    print(daf.shape)
    print(daf.head())
    print(daf.head(10))
    print(daf.target.value_counts())
    print(daf.target.value_counts(normalize=True))
    daf.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"])
    print(daf.info())
    print(daf.describe())
    print(daf.sex.value_counts())
    print(crosstab(daf.target, daf.sex))
    crosstab(daf.target, daf.sex).plot(
        kind="bar", figsize=(10, 6), color=["salmon", "lightblue"]
    )
    crosstab(daf.target, daf.sex).plot(
        kind="bar", figsize=(10, 6), color=["salmon", "lightblue"]
    )
    title("Heart Disease Frequency for Sex")
    xlabel("0 = No Disease, 1 = Disease")
    ylabel("Amount")
    legend(["Female", "Male"])
    xticks(rotation=0)
    figure(figsize=(10, 6))
    scatter(daf.age[daf.target == 1], daf.thalach[daf.target == 1], c="salmon")
    scatter(daf.age[daf.target == 0], daf.thalach[daf.target == 0], c="lightblue")
    title("Heart Disease in function of Age and Max Heart Rate")
    xlabel("Age")
    legend(["Disease", "No Disease"])
    ylabel("Max Heart Rate")
    figure()
    daf.age.plot.hist()
    print(crosstab(daf.cp, daf.target))
    crosstab(daf.cp, daf.target).plot(
        kind="bar", figsize=(10, 6), color=["lightblue", "salmon"]
    )
    title("Heart Disease Frequency Per Chest Pain Type")
    xlabel("Chest Pain Type")
    ylabel("Frequency")
    legend(["No Disease", "Disease"])
    xticks(rotation=0)
    corr_matrix = daf.corr()
    print(corr_matrix)
    corr_matrix = daf.corr()

    corr(corr_matrix, daf)


def corr(corr_matrix, daf):
    """
    Correlation between independent variables
    """
    figure(figsize=(15, 10))
    heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
    print(daf.head())
    xda = daf.drop("target", axis=1)
    yda = daf.target.values
    print(xda.head())
    print(yda)
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.2)
    print(x_train.head())
    print(y_train, len(y_train))
    print(x_train.head())
    print(y_test, len(y_test))
    models = {
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
    }
    model_scores = fit_and_score(
        models=models, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test
    )
    print(model_scores)
    model_compare = DataFrame(model_scores, index=["accuracy"])
    model_compare.T.plot.bar()
    train_scores = []
    test_scores = []
    neighbors = range(1, 21)
    knn = KNeighborsClassifier()
    for i in neighbors:
        knn.set_params(n_neighbors=i)
        knn.fit(x_train, y_train)
        train_scores.append(knn.score(x_train, y_train))
        test_scores.append(knn.score(x_test, y_test))
    print(train_scores)
    figure()
    plot(neighbors, train_scores, label="Train score")
    plot(neighbors, test_scores, label="Test score")
    xticks(arange(1, 21, 1))
    xlabel("Number of neighbors")
    ylabel("Model score")
    legend()
    print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")

    data = SimpleNamespace(
        xda=xda,
        yda=yda,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        daf=daf,
    )
    hyperparam(data)


def hyperparam(data):
    """
    Different LogisticRegression hyperparameters
    Different RandomForestClassifier hyperparameters
    """
    log_reg_grid = {"C": logspace(-4, 4, 20), "solver": ["liblinear"]}
    rf_grid = {
        "n_estimators": arange(10, 1000, 50),
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": arange(2, 20, 2),
        "min_samples_leaf": arange(1, 20, 2),
    }
    rs_log_reg = RandomizedSearchCV(
        LogisticRegression(),
        param_distributions=log_reg_grid,
        cv=5,
        n_iter=20,
        verbose=True,
    )
    rs_log_reg.fit(data.x_train, data.y_train)
    print(rs_log_reg.best_params_)
    print(rs_log_reg.score(data.x_test, data.y_test))
    rs_rf = RandomizedSearchCV(
        RandomForestClassifier(),
        param_distributions=rf_grid,
        cv=5,
        n_iter=20,
        verbose=True,
    )
    rs_rf.fit(data.x_train, data.y_train)
    print(rs_rf.best_params_)
    print(rs_rf.score(data.x_test, data.y_test))
    log_reg_grid = {"C": logspace(-4, 4, 20), "solver": ["liblinear"]}
    gs_log_reg = GridSearchCV(
        LogisticRegression(), param_grid=log_reg_grid, cv=5, verbose=True
    )
    gs_log_reg.fit(data.x_train, data.y_train)
    print(gs_log_reg.best_params_)
    print(gs_log_reg.score(data.x_test, data.y_test))
    y_preds = gs_log_reg.predict(data.x_test)
    print(y_preds)
    print(data.y_test)
    plot_roc_curve(gs_log_reg, data.x_test, data.y_test)
    print(confusion_matrix(data.y_test, y_preds))
    sset(font_scale=1.5)
    plot_conf_mat(data.y_test, y_preds)
    print(classification_report(data.y_test, y_preds))
    print(gs_log_reg.best_params_)
    clf = LogisticRegression(C=0.23357214690901212, solver="liblinear")
    cv_acc = cross_val_score(clf, data.xda, data.yda, cv=5, scoring="accuracy")
    print(cv_acc)
    cv_acc = mean(cv_acc)
    print(cv_acc)
    cv_precision = mean(
        cross_val_score(clf, data.xda, data.yda, cv=5, scoring="precision")
    )
    print(cv_precision)
    cv_recall = mean(cross_val_score(clf, data.xda, data.yda, cv=5, scoring="recall"))
    print(cv_recall)
    cv_f1 = mean(cross_val_score(clf, data.xda, data.yda, cv=5, scoring="f1"))
    print(cv_f1)
    cv_metrics = DataFrame(
        {
            "Accuracy": cv_acc,
            "Precision": cv_precision,
            "Recall": cv_recall,
            "F1": cv_f1,
        },
        index=[0],
    )
    cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False)
    clf.fit(data.x_train, data.y_train)
    print(clf.coef_)
    features_dict = dict(zip(data.daf.columns, list(clf.coef_[0])))
    print(features_dict)
    features_df = DataFrame(features_dict, index=[0])
    features_df.T.plot.bar(title="Feature Importance", legend=False)
    print(crosstab(data.daf["sex"], data.daf["target"]))
    print(crosstab(data.daf["slope"], data.daf["target"]))


if __name__ == "__main__":
    wrapper()
    show()
