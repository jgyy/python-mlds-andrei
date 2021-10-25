"""
Matplotlib Practice
"""
from os.path import join, dirname
from matplotlib.pyplot import plot, figure, show, subplots, style
from numpy import linspace, exp, sin
from numpy.random import randn, random, rand
from pandas import DataFrame, read_csv, date_range

file = lambda x: join(dirname(__file__), x)


def wrapper():
    """
    wrapper function
    """
    figure()
    plot()
    figure()
    plot([2, 6, 8, 17])
    xda = [22, 88, 98, 103, 45]
    yda = [7, 8, 9, 10, 11]
    figure()
    plot(xda, yda)
    subplots()
    _, axi = subplots()
    axi.plot(xda, yda)
    xda = [34, 77, 21, 54, 9]
    yda = [9, 45, 89, 66, 4]
    fig, axi = subplots()
    axi.plot(xda, yda)
    axi.set(title="Sample simple plot", xlabel="x-axis", ylabel="y-axis")
    fig.savefig(file("simple-plot.png"))
    xda = linspace(0, 10, 100)
    _, axi = subplots()
    axi.plot(xda, xda ** 2)
    _, axi = subplots()
    axi.scatter(xda, exp(xda))
    _, axi = subplots()
    axi.scatter(xda, sin(xda))
    favourite_food_prices = {"Almond butter": 10, "Blueberries": 5, "Eggs": 6}
    _, axi = subplots()
    axi.bar(favourite_food_prices.keys(), favourite_food_prices.values())
    axi.set(title="Daniel's favourite foods", xlabel="Food", ylabel="Price ($)")
    _, axi = subplots()
    axi.barh(list(favourite_food_prices.keys()), list(favourite_food_prices.values()))
    axi.set(title="Daniel's favourite foods", xlabel="Price ($)", ylabel="Food")
    xda = randn(1000)
    _, axi = subplots()
    axi.hist(xda)
    xda = random(1000)
    _, axi = subplots()
    axi.hist(xda)
    subplots(nrows=2, ncols=2)
    _, ((axi, ax2), (ax3, ax4)) = subplots(nrows=2, ncols=2, figsize=(10, 5))
    axi.plot(xda, xda / 2)
    ax2.scatter(random(10), random(10))
    ax3.bar(favourite_food_prices.keys(), favourite_food_prices.values())
    ax4.hist(randn(1000))

    carsale()


def carsale():
    """
    car sales function
    """
    car_sales = DataFrame(read_csv(file("car-sales.csv")))
    print(car_sales)
    car_sales["Price"] = car_sales["Price"].str.replace(r"[\$\,\.]", "", regex=True)
    car_sales["Price"] = car_sales["Price"].str[:-2]
    car_sales["Total Sales"] = car_sales["Price"].astype(int).cumsum()
    car_sales["Sale Date"] = date_range("13/1/2020", periods=len(car_sales))
    print(car_sales)
    car_sales.plot(x="Sale Date", y="Total Sales")
    car_sales["Price"] = car_sales["Price"].astype(int)
    car_sales.plot(x="Odometer (KM)", y="Price", kind="scatter")
    xda = rand(10, 4)
    daf = DataFrame(xda, columns=["a", "b", "c", "d"])
    daf.plot(kind="bar")
    car_sales.plot(x="Make", y="Odometer (KM)", kind="bar")
    car_sales["Odometer (KM)"].plot(kind="hist")
    car_sales["Price"].plot.hist(bins=20)
    heart_disease = DataFrame(read_csv(file("heart-disease.csv")))
    print(heart_disease.head(10))
    heart_disease["age"].plot.hist(bins=50)
    heart_disease.plot.hist(subplots=True)
    heart_disease.plot.hist(figsize=(10, 30), subplots=True)
    over_50 = heart_disease[heart_disease["age"] > 50]
    _, axi = subplots(figsize=(10, 6))
    scatter = axi.scatter(over_50["age"], over_50["chol"], c=over_50["target"])
    axi.set(
        title="Heart Disease and Cholesterol Levels", xlabel="Age", ylabel="Cholesterol"
    )
    axi.legend(*scatter.legend_elements(), title="Target")
    axi.axhline(over_50["chol"].mean(), linestyle="--")
    print(style.available)
    style.use("seaborn-whitegrid")
    _, axi = subplots(figsize=(10, 6))
    scatter = axi.scatter(over_50["age"], over_50["chol"], c=over_50["target"])
    axi.set(
        title="Heart Disease and Cholesterol Levels", xlabel="Age", ylabel="Cholesterol"
    )
    axi.legend(*scatter.legend_elements(), title="Target")
    axi.axhline(over_50["chol"].mean(), linestyle="--")
    fig, axi = subplots(figsize=(10, 6))
    scatter = axi.scatter(
        over_50["age"], over_50["chol"], c=over_50["target"], cmap="winter"
    )
    axi.set(
        title="Heart Disease and Cholesterol Levels", xlabel="Age", ylabel="Cholesterol"
    )
    axi.legend(*scatter.legend_elements(), title="Target")
    axi.axhline(over_50["chol"].mean(), linestyle="--", color="red")
    fig.savefig(file("matplotlib-heart-disease-chol-age-plot-saved.png"))
    subplots()


if __name__ == "__main__":
    wrapper()
    show()
