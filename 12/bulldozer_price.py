"""
🚜 Predicting the Sale Price of Bulldozers using Machine Learning model
"""
from os.path import join, dirname
from numpy import sqrt, arange
from matplotlib.pyplot import subplots, show, figure
from seaborn import barplot
from pandas import DataFrame, read_csv, isnull, Categorical
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

readcsv = lambda x, **y: DataFrame(read_csv(join(dirname(__file__), x), **y))
tocsv = lambda x, y, **z: x.to_csv(join(dirname(__file__), y), **z)


def rmsle(y_test, y_preds):
    """
    Create evaluation function (the competition uses Root Mean Square Log Error)
    """
    return sqrt(mean_squared_log_error(y_test, y_preds))


def show_scores(model, x_train, x_valid, y_train, y_valid):
    """
    Create function to evaluate our model
    """
    train_preds = model.predict(x_train)
    val_preds = model.predict(x_valid)
    scores = {
        "Training MAE": mean_absolute_error(y_train, train_preds),
        "Valid MAE": mean_absolute_error(y_valid, val_preds),
        "Training RMSLE": rmsle(y_train, train_preds),
        "Valid RMSLE": rmsle(y_valid, val_preds),
        "Training R^2": model.score(x_train, y_train),
        "Valid R^2": model.score(x_valid, y_valid),
    }
    return scores


def preprocess_data(daf):
    """
    Add datetime parameters for saledate > Drop original saledate
    Fill numeric rows with the median > Turn categorical variables into numbers
    Add the +1 because pandas encodes missing categories as -1
    """
    daf["saleYear"] = daf.saledate.dt.year
    daf["saleMonth"] = daf.saledate.dt.month
    daf["saleDay"] = daf.saledate.dt.day
    daf["saleDayofweek"] = daf.saledate.dt.dayofweek
    daf["saleDayofyear"] = daf.saledate.dt.dayofyear
    daf.drop("saledate", axis=1, inplace=True)
    for label, content in daf.items():
        if is_numeric_dtype(content):
            if isnull(content).sum():
                daf[label + "_is_missing"] = isnull(content)
                daf[label] = content.fillna(content.median())
        if not is_numeric_dtype(content):
            daf[label + "_is_missing"] = isnull(content)
            daf[label] = Categorical(content).codes + 1
    return daf


def plot_features(columns, importances, num=20):
    """
    Helper function for plotting feature importance
    """
    daf = (
        DataFrame({"features": columns, "feature_importance": importances})
        .sort_values("feature_importance", ascending=False)
        .reset_index(drop=True)
    )
    figure()
    barplot(x="feature_importance", y="features", data=daf[:num], orient="h")


def wrapper():
    """
    wrapper function
    """
    daf = readcsv("TrainAndValid.csv")
    print(daf.info())
    _, axi = subplots()
    axi.scatter(daf["saledate"][:1000], daf["SalePrice"][:1000])
    figure()
    daf.SalePrice.plot.hist()
    daf = readcsv("TrainAndValid.csv", low_memory=False, parse_dates=["saledate"])
    print(daf.info())
    _, axi = subplots()
    axi.scatter(daf["saledate"][:1000], daf["SalePrice"][:1000])
    print(daf.head())
    print(daf.head().T)
    print(daf.saledate.head(20))
    daf.sort_values(by=["saledate"], inplace=True, ascending=True)
    print(daf.saledate.head(20))
    df_tmp = daf.copy()
    df_tmp["saleYear"] = df_tmp.saledate.dt.year
    df_tmp["saleMonth"] = df_tmp.saledate.dt.month
    df_tmp["saleDay"] = df_tmp.saledate.dt.day
    df_tmp["saleDayofweek"] = df_tmp.saledate.dt.dayofweek
    df_tmp["saleDayofyear"] = df_tmp.saledate.dt.dayofyear
    df_tmp.drop("saledate", axis=1, inplace=True)
    print(df_tmp.head().T)
    print(df_tmp.state.value_counts())
    print(df_tmp.info())
    print(df_tmp.isna().sum())
    print(df_tmp.head().T)
    print(is_string_dtype(df_tmp["UsageBand"]))
    for label, content in df_tmp.items():
        if is_string_dtype(content):
            print(label)
    random_dict = {"key1": "hello", "key2": "world!"}
    for key, value in random_dict.items():
        print(f"This is a key: {key}")
        print(f"This is a value: {value}")
    for label, content in df_tmp.items():
        if is_string_dtype(content):
            df_tmp[label] = content.astype("category").cat.as_ordered()
    print(df_tmp.info())
    print(df_tmp.state.cat.categories)
    print(df_tmp.state.cat.codes)
    print(df_tmp.isnull().sum() / len(df_tmp))
    tocsv(df_tmp, "train_tmp.csv", index=False)
    df_tmp = readcsv("train_tmp.csv", low_memory=False)
    print(df_tmp.head().T)
    print(df_tmp.isna().sum())

    fill_num(df_tmp, daf)


def fill_num(df_tmp, daf):
    """
    Fill missing values function
    """
    for label, content in df_tmp.items():
        if is_numeric_dtype(content):
            print(label)
    for label, content in df_tmp.items():
        if is_numeric_dtype(content):
            if isnull(content).sum():
                df_tmp[label + "_is_missing"] = isnull(content)
                df_tmp[label] = content.fillna(content.median())
    for label, content in df_tmp.items():
        if is_numeric_dtype(content):
            if isnull(content).sum():
                print(label)
    print(df_tmp.auctioneerID_is_missing.value_counts())
    for label, content in df_tmp.items():
        if not is_numeric_dtype(content):
            print(label)
    for label, content in df_tmp.items():
        if not is_numeric_dtype(content):
            df_tmp[label + "_is_missing"] = isnull(content)
            df_tmp[label] = Categorical(content).codes + 1
    print(df_tmp.info())
    print(df_tmp.isna().sum())
    print(df_tmp.head().T)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(df_tmp.drop("SalePrice", axis=1), df_tmp.SalePrice)
    print(model.score(df_tmp.drop("SalePrice", axis=1), df_tmp.SalePrice))
    print(df_tmp.head())
    print(df_tmp.saleYear.value_counts())
    df_val = df_tmp[df_tmp.saleYear == 2012]
    df_train = df_tmp[df_tmp.saleYear != 2012]
    print(len(df_val), len(df_train))
    x_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
    x_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice
    print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
    print(len(x_train))
    model = RandomForestRegressor(n_jobs=-1, max_samples=10000)
    model.fit(x_train, y_train)
    print(show_scores(model, x_train, x_valid, y_train, y_valid))

    hyperparam(x_train, x_valid, y_train, y_valid, daf)


def hyperparam(x_train, x_valid, y_train, y_valid, daf):
    """
    Hyperparameter tuning with RandomizedSearchCV function
    """
    rf_grid = {
        "n_estimators": arange(10, 100, 10),
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": arange(2, 20, 2),
        "min_samples_leaf": arange(1, 20, 2),
        "max_features": [0.5, 1, "sqrt", "auto"],
        "max_samples": [10000],
    }
    rs_model = RandomizedSearchCV(
        RandomForestRegressor(),
        param_distributions=rf_grid,
        n_iter=20,
        cv=5,
        verbose=True,
    )
    rs_model.fit(x_train, y_train)
    print(rs_model.best_params_)
    print(show_scores(rs_model, x_train, x_valid, y_train, y_valid))
    ideal_model = RandomForestRegressor(
        n_estimators=90,
        min_samples_leaf=1,
        min_samples_split=14,
        max_features=0.5,
        n_jobs=-1,
        max_samples=None,
    )
    ideal_model.fit(x_train, y_train)
    print(show_scores(ideal_model, x_train, x_valid, y_train, y_valid))
    fast_model = RandomForestRegressor(
        n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1
    )
    fast_model.fit(x_train, y_train)
    print(show_scores(fast_model, x_train, x_valid, y_train, y_valid))
    df_test = readcsv("Test.csv", parse_dates=["saledate"])
    print(df_test.head())
    df_test = preprocess_data(df_test)
    print(df_test.head())
    print(x_train.head())
    for i in set(x_train.columns) - set(df_test.columns):
        print(i)
        df_test[i] = False
    print(df_test.head())
    test_preds = ideal_model.predict(df_test)
    df_preds = DataFrame()
    df_preds["SalesID"] = df_test["SalesID"]
    df_preds["SalePrice"] = test_preds
    print(df_preds)
    tocsv(df_preds, "predictions.csv", index=False)
    print(ideal_model.feature_importances_)
    plot_features(x_train.columns, ideal_model.feature_importances_)
    print(sum(ideal_model.feature_importances_))
    print(daf.ProductSize.isna().sum())
    print(daf.ProductSize.value_counts())
    print(daf.Turbocharged.value_counts())
    print(daf.Thumb.value_counts())


if __name__ == "__main__":
    wrapper()
    show()
