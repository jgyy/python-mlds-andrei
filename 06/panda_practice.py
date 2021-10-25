"""
Pandas Practice
"""
from os.path import join, dirname
from pandas import Series, DataFrame, read_csv, crosstab
from matplotlib.pyplot import figure, show

csv = lambda x: join(dirname(__file__), x)


def wrapper():
    """
    wrapper function
    """
    colours = Series(["Blue", "Red", "White"])
    print(colours)
    cars = Series(["BMW", "Toyota", "Honda"])
    print(cars)
    car_data = DataFrame({"Car make": cars, "Colour": colours})
    print(car_data)
    car_sales = DataFrame(read_csv(csv("car-sales.csv")))
    print(car_sales)
    car_sales.to_csv(csv("exported-car-sales.csv"))
    print(car_sales.dtypes)
    print(car_sales.describe())
    print(car_sales.info())
    series = Series([999, 22203, 43920])
    print(series.mean())
    series = Series([482392, 34994, 22])
    print(series.sum())
    print(car_sales.columns)
    print(len(car_sales))
    print(car_sales.head())
    print(car_sales.head(7))
    print(car_sales.tail())
    print(car_sales.loc[3])
    print(car_sales.iloc[3])
    print(car_sales["Odometer (KM)"])
    print(car_sales["Odometer (KM)"].mean())
    print(car_sales[car_sales["Odometer (KM)"] > 100000])
    print(crosstab(car_sales["Make"], car_sales["Doors"]))
    print(car_sales.groupby(["Make"]).mean())
    figure()
    car_sales["Odometer (KM)"].plot()
    figure()
    car_sales["Odometer (KM)"].hist()
    car_sales["Price"] = car_sales["Price"].str.replace("[\$\,\.]", "")
    print(car_sales["Price"])
    car_sales["Price"] = car_sales["Price"].str[:-2]
    print(car_sales["Price"])
    car_sales["Price"] = car_sales["Price"].astype(int)
    print(car_sales["Make"].str.lower())
    car_sales["Make"] = car_sales["Make"].str.lower()
    print(car_sales)
    car_sales_missing = DataFrame(read_csv(csv("car-sales-missing-data.csv")))
    print(car_sales_missing)
    car_sales_missing["Odometer"].fillna(
        car_sales_missing["Odometer"].mean(), inplace=True
    )

    carsale(car_sales_missing, car_sales)


def carsale(car_sales_missing, car_sales):
    """
    car sales function
    """
    print(car_sales_missing)
    car_sales_missing.dropna(inplace=True)
    print(car_sales_missing)
    car_sales["Seats"] = 5
    print(car_sales)
    engine_sizes = [1.3, 4.3, 2.3, 3.3, 3.0, 2.3, 1.4, 1.7, 2.5, 3.1]
    car_sales["Engine Size"] = engine_sizes
    print(car_sales)
    car_sales["Price per KM"] = car_sales["Price"] / car_sales["Odometer (KM)"]
    print(car_sales)
    car_sales = car_sales.drop("Price per KM", axis=1)
    print(car_sales)
    car_sales_sampled = car_sales_sampled = car_sales.sample(frac=1)
    print(car_sales_sampled)
    print(car_sales_sampled.reset_index())
    car_sales["Odometer (KM)"] = car_sales["Odometer (KM)"].apply(lambda x: x / 1.6)
    print(car_sales)
    car_sales = car_sales.rename(columns={"Odometer (KM)": "Odometer (Miles)"})
    print(car_sales)


if __name__ == "__main__":
    wrapper()
    show()
