"""
Introduction to Matplotlib
"""
from os.path import join, dirname
from matplotlib.pyplot import plot, figure, show, subplots, style
from numpy import linspace, exp, sin
from numpy.random import randn, random, rand
from pandas import DataFrame, Series, read_csv, date_range

file = lambda x: join(dirname(__file__), x)


def wrapper():
    """
    wrapper function
    """
    figure()
    plot()
    figure()
    plot([1, 2, 3, 4])
    xda = [1, 2, 3, 4]
    yda = [11, 22, 33, 44]
    figure()
    plot(xda, yda)
    _ = figure()
    _.add_subplot()
    _, axi = subplots()
    axi.plot(xda, yda)
    print(type(_), type(axi))
    _, axi = subplots(figsize=(10, 10))
    axi.plot(xda, yda)
    axi.set(title="Sample Simple Plot", xlabel="x-axis", ylabel="y-axis")
    _.savefig(file("simple-plot.png"))
    xda = linspace(0, 10, 100)
    print(xda[:10])
    xda = linspace(0, 10, 100)
    print(xda[:10])
    _, axi = subplots()
    axi.plot(xda, xda ** 2)
    _, axi = subplots()
    axi.scatter(xda, exp(xda))
    _, axi = subplots()
    axi.scatter(xda, sin(xda))
    nut_butter_prices = {"Almond butter": 10, "Peanut butter": 8, "Cashew butter": 12}
    _, axi = subplots()
    axi.bar(nut_butter_prices.keys(), nut_butter_prices.values())
    axi.set(title="Dan's Nut Butter Store", ylabel="Price ($)")
    _, axi = subplots()
    axi.barh(list(nut_butter_prices.keys()), list(nut_butter_prices.values()))
    xda = randn(1000)
    _, axi = subplots()
    axi.hist(xda)
    xda = random(1000)
    _, axi = subplots()
    axi.hist(xda)
    _, ((ax1, ax2), (ax3, ax4)) = subplots(nrows=2, ncols=2, figsize=(10, 5))
    ax1.plot(xda, xda / 2)
    ax2.scatter(random(10), random(10))
    ax3.bar(nut_butter_prices.keys(), nut_butter_prices.values())
    ax4.hist(randn(1000))

    subplots_func(xda, nut_butter_prices)

def subplots_func(xda, nut_butter_prices):
    """
    subplots functions
    """
    _, axi = subplots(nrows=2, ncols=2, figsize=(10, 5))
    axi[0, 0].plot(xda, xda / 2)
    axi[0, 1].scatter(random(10), random(10))
    axi[1, 0].bar(nut_butter_prices.keys(), nut_butter_prices.values())
    axi[1, 1].hist(randn(1000))
    car_sales = DataFrame(read_csv(file("car-sales.csv")))
    print(car_sales)
    tes = Series(randn(1000), index=date_range("1/1/2020", periods=1000))
    print(tes)
    print(tes.cumsum())
    figure()
    tes.cumsum().plot()
    car_sales["Price"] = car_sales["Price"].str.replace(r"[\$\,\.]", "", regex=True)
    print(car_sales)
    car_sales["Price"] = car_sales["Price"].str[:-2]
    print(car_sales)
    car_sales["Sale Date"] = date_range("1/1/2020", periods=len(car_sales))
    print(car_sales)
    car_sales["Total Sales"] = car_sales["Price"].astype(int).cumsum()
    print(car_sales)
    figure()
    car_sales.plot(x="Sale Date", y="Total Sales")
    car_sales["Price"] = car_sales["Price"].astype(int)
    car_sales.plot(x="Odometer (KM)", y="Price", kind="scatter")
    xda = rand(10, 4)
    print(xda)
    daf = DataFrame(xda, columns=["a", "b", "c", "d"])
    print(daf)
    daf.plot.bar()
    car_sales.plot(x="Make", y="Odometer (KM)", kind="bar")
    figure()
    car_sales["Odometer (KM)"].plot.hist()
    figure()
    car_sales["Odometer (KM)"].plot.hist(bins=20)
    figure()
    car_sales["Price"].plot.hist(bins=10)

    heart_dataset(car_sales)


def heart_dataset(car_sales):
    """
    heart disease dataset function
    """
    heart_disease = DataFrame(read_csv(file("heart-disease.csv")))
    print(heart_disease.head())
    figure()
    heart_disease["age"].plot.hist(bins=50)
    heart_disease.plot.hist(figsize=(10, 30), subplots=True)
    over_50 = heart_disease[heart_disease["age"] > 50]
    print(over_50)
    over_50.plot(kind="scatter", x="age", y="chol", c="target", figsize=(10, 6))
    _, axi = subplots(figsize=(10, 6))
    over_50.plot(kind="scatter", x="age", y="chol", c="target", ax=axi)
    axi.set_xlim([45, 100])
    _, axi = subplots(figsize=(10, 6))
    scatter = axi.scatter(over_50["age"], over_50["chol"], c=over_50["target"])
    axi.set(
        title="Heart Disease and Cholesterol Levels", xlabel="Age", ylabel="Cholesterol"
    )
    axi.legend(*scatter.legend_elements(), title="Target")
    _, axi = subplots(figsize=(10, 6))
    scatter = axi.scatter(over_50["age"], over_50["chol"], c=over_50["target"])
    axi.set(
        title="Heart Disease and Cholesterol Levels", xlabel="Age", ylabel="Cholesterol"
    )
    axi.legend(*scatter.legend_elements(), title="Target")
    axi.axhline(over_50["chol"].mean(), linestyle="--")
    _, (ax0, ax1) = subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))
    scatter = ax0.scatter(over_50["age"], over_50["chol"], c=over_50["target"])
    ax0.set(title="Heart Disease and Cholesterol Levels", ylabel="Cholesterol")
    ax0.legend(*scatter.legend_elements(), title="Target")
    ax0.axhline(y=over_50["chol"].mean(), color="b", linestyle="--", label="Average")
    scatter = ax1.scatter(over_50["age"], over_50["thalach"], c=over_50["target"])
    ax1.set(
        title="Heart Disease and Max Heart Rate Levels",
        xlabel="Age",
        ylabel="Max Heart Rate",
    )
    ax1.legend(*scatter.legend_elements(), title="Target")
    ax1.axhline(y=over_50["thalach"].mean(), color="b", linestyle="--", label="Average")
    _.suptitle("Heart Disease Analysis", fontsize=16, fontweight="bold")
    print(style.available)

    carsales(car_sales, over_50)


def carsales(car_sales, over_50):
    """
    car sales function
    """
    figure()
    car_sales["Price"].plot()
    figure()
    style.use("seaborn-whitegrid")
    car_sales["Price"].plot()
    figure()
    style.use("seaborn")
    car_sales["Price"].plot()
    car_sales.plot(x="Odometer (KM)", y="Price", kind="scatter")
    figure()
    style.use("ggplot")
    car_sales["Price"].plot.hist()
    xda = randn(10, 4)
    print(xda)
    daf = DataFrame(xda, columns=["a", "b", "c", "d"])
    print(daf)
    figure()
    axi = plot(kind="bar")
    print(type(axi))
    axi = daf.plot(figsize=(10, 6), kind="bar")
    axi.set(
        title="Random Number Bar Graph from DataFrame",
        xlabel="Row number",
        ylabel="Random number",
    )
    axi.legend().set_visible(True)
    style.use("seaborn-whitegrid")
    _, axi = subplots(figsize=(10, 6))
    scatter = axi.scatter(over_50["age"], over_50["chol"], c=over_50["target"])
    axi.set(
        title="Heart Disease and Cholesterol Levels", xlabel="Age", ylabel="Cholesterol"
    )
    axi.axhline(y=over_50["chol"].mean(), c="b", linestyle="--", label="Average")
    axi.legend(*scatter.legend_elements(), title="Target")
    _, axi = subplots(figsize=(10, 6))
    scatter = axi.scatter(
        over_50["age"], over_50["chol"], c=over_50["target"], cmap="winter"
    )
    axi.set(
        title="Heart Disease and Cholesterol Levels", xlabel="Age", ylabel="Cholesterol"
    )
    axi.axhline(y=over_50["chol"].mean(), color="r", linestyle="--", label="Average")
    axi.legend(*scatter.legend_elements(), title="Target")

    heart_subplots(over_50)


def heart_subplots(over_50):
    """
    heart subplots function
    """
    fig, (ax0, ax1) = subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))
    scatter = ax0.scatter(
        over_50["age"], over_50["chol"], c=over_50["target"], cmap="winter"
    )
    ax0.set(title="Heart Disease and Cholesterol Levels", ylabel="Cholesterol")
    ax0.axhline(y=over_50["chol"].mean(), color="r", linestyle="--", label="Average")
    ax0.legend(*scatter.legend_elements(), title="Target")
    scatter = ax1.scatter(
        over_50["age"], over_50["thalach"], c=over_50["target"], cmap="winter"
    )
    ax1.set(
        title="Heart Disease and Max Heart Rate Levels",
        xlabel="Age",
        ylabel="Max Heart Rate",
    )
    ax1.axhline(y=over_50["thalach"].mean(), color="r", linestyle="--", label="Average")
    ax1.legend(*scatter.legend_elements(), title="Target")
    fig.suptitle("Heart Disease Analysis", fontsize=16, fontweight="bold")
    fig, (ax0, ax1) = subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))
    scatter = ax0.scatter(
        over_50["age"], over_50["chol"], c=over_50["target"], cmap="winter"
    )
    ax0.set(title="Heart Disease and Cholesterol Levels", ylabel="Cholesterol")
    ax0.set_xlim([50, 80])
    ax0.axhline(y=over_50["chol"].mean(), color="r", linestyle="--", label="Average")
    ax0.legend(*scatter.legend_elements(), title="Target")
    scatter = ax1.scatter(
        over_50["age"], over_50["thalach"], c=over_50["target"], cmap="winter"
    )
    ax1.set(
        title="Heart Disease and Max Heart Rate Levels",
        xlabel="Age",
        ylabel="Max Heart Rate",
    )
    ax1.set_ylim([60, 200])
    ax1.axhline(y=over_50["thalach"].mean(), color="r", linestyle="--", label="Average")
    ax1.legend(*scatter.legend_elements(), title="Target")
    fig.suptitle("Heart Disease Analysis", fontsize=16, fontweight="bold")
    fig, (ax0, ax1) = subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))
    scatter = ax0.scatter(
        over_50["age"], over_50["chol"], c=over_50["target"], cmap="winter"
    )
    ax0.set(title="Heart Disease and Cholesterol Levels", ylabel="Cholesterol")
    ax0.set_xlim([50, 80])
    ax0.axhline(y=over_50["chol"].mean(), color="r", linestyle="--", label="Average")
    ax0.legend(*scatter.legend_elements(), title="Target")
    scatter = ax1.scatter(
        over_50["age"], over_50["thalach"], c=over_50["target"], cmap="winter"
    )
    ax1.set(
        title="Heart Disease and Max Heart Rate Levels",
        xlabel="Age",
        ylabel="Max Heart Rate",
    )
    ax1.set_ylim([60, 200])
    ax1.axhline(y=over_50["thalach"].mean(), color="r", linestyle="--", label="Average")
    ax1.legend(*scatter.legend_elements(), title="Target")
    fig.suptitle("Heart Disease Analysis", fontsize=16, fontweight="bold")
    print(fig.canvas.get_supported_filetypes())
    fig.savefig(file("heart-disease-analysis.png"))



if __name__ == "__main__":
    wrapper()
    show()
