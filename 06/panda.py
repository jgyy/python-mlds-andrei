"""
Introduction to Pandas
"""
from os.path import join, dirname
from pandas import Series, DataFrame, read_csv, crosstab
from matplotlib.pyplot import figure, show

csv = lambda x: join(dirname(__file__), x)


def wrapper():
    """
    wrapper function
    """
    series = Series(["BMW", "Toyota", "Honda"])
    print(series)
    colours = Series(["Red", "Blue", "White"])
    print(colours)
    car_data = DataFrame({"Car make": series, "Colour": colours})
    print(car_data)
    car_sales = DataFrame(read_csv(csv("car-sales.csv")))
    print(car_sales)
    car_sales.to_csv(csv("exported-car-sales.csv"), index=False)
    exported_car_sales = DataFrame(read_csv(csv("exported-car-sales.csv")))
    print(exported_car_sales)
    print(car_sales.dtypes)
    print(car_sales.columns)
    car_columns = car_sales.columns
    print(car_columns)
    print(car_sales.index)
    print(car_sales)
    print(car_sales.describe())
    print(car_sales.info())
    print(car_sales.mean())
    car_prices = Series([3000, 1500, 111250])
    print(car_prices.mean())
    print(car_sales.sum())
    print(car_sales["Doors"].sum())
    print(len(car_sales))
    print(car_sales)
    print(car_sales.head())
    print(car_sales)
    print(car_sales.head(7))
    print(car_sales.tail(3))
    animals = Series(["cat", "dog", "bird", "panda", "snake"], index=[0, 3, 9, 8, 3])
    print(animals)
    print(animals.loc[3])
    print(animals.loc[9])
    print(car_sales.loc[3])
    print(animals)
    print(animals.iloc[3])
    print(car_sales.loc[3])
    print(car_sales)
    print(animals)
    print(animals.iloc[:3])
    print(car_sales.loc[:3])
    print(car_sales.head(4))

    carsale_func(car_sales)


def carsale_func(car_sales):
    """
    car sales function
    """
    print(car_sales["Make"])
    print(car_sales["Colour"])
    print(car_sales["Odometer (KM)"])
    print(car_sales[car_sales["Make"] == "Toyota"])
    print(car_sales[car_sales["Odometer (KM)"] > 100000])
    print(car_sales)
    print(crosstab(car_sales["Make"], car_sales["Doors"]))
    print(car_sales.groupby(["Make"]).mean())
    figure()
    car_sales["Odometer (KM)"].plot()
    figure()
    car_sales["Odometer (KM)"].hist()
    print(car_sales["Price"].dtype)
    print(car_sales)
    car_sales["Price"] = car_sales["Price"].str.replace(r"[\$\,\.]", "").astype(int)
    print(car_sales)
    figure()
    car_sales["Price"].plot()

    manipulate(car_sales)


def manipulate(car_sales):
    """
    manipulating data function
    """
    print(car_sales["Make"].str.lower())
    print(car_sales)
    car_sales["Make"] = car_sales["Make"].str.lower()
    print(car_sales)
    car_sales_missing = DataFrame(read_csv(csv("car-sales-missing-data.csv")))
    print(car_sales_missing)
    print(car_sales_missing["Odometer"].mean())
    print(car_sales_missing["Odometer"].fillna(car_sales_missing["Odometer"].mean()))
    print(car_sales_missing)
    car_sales_missing["Odometer"].fillna(
        car_sales_missing["Odometer"].mean(), inplace=True
    )
    print(car_sales_missing)
    print(car_sales_missing.dropna())
    car_sales_missing.dropna(inplace=True)
    print(car_sales_missing)
    car_sales_missing = DataFrame(read_csv(csv("car-sales-missing-data.csv")))
    print(car_sales_missing)
    car_sales_missing_dropped = car_sales_missing.dropna()
    print(car_sales_missing_dropped)
    car_sales_missing_dropped.to_csv(csv("car-sales-missing-dropped.csv"))
    print(car_sales)
    seats_column = Series([5, 5, 5, 5, 5])
    car_sales["Seats"] = seats_column
    print(car_sales)
    car_sales["Seats"].fillna(5, inplace=True)
    print(car_sales)
    fuel_economy = [7.5, 9.2, 5.0, 9.6, 8.7, 4.7, 7.6, 8.7, 3.0, 4.5]
    car_sales["Fuel per 100KM"] = fuel_economy
    print(car_sales)
    car_sales["Total fuel used (L)"] = (
        car_sales["Odometer (KM)"] / 100 * car_sales["Fuel per 100KM"]
    )
    print(car_sales)
    car_sales["Number of wheels"] = 4
    print(car_sales)
    car_sales["Passed road saftey"] = True
    print(car_sales.dtypes)
    print(car_sales)
    car_sales_shuffled = car_sales.sample(frac=1)
    print(car_sales_shuffled)
    print(car_sales_shuffled.sample(frac=0.01))
    print(car_sales_shuffled)
    car_sales_shuffled.reset_index(drop=True, inplace=True)
    print(car_sales_shuffled)
    print(car_sales)
    car_sales["Odometer (KM)"] = car_sales["Odometer (KM)"].apply(lambda x: x / 1.6)
    print(car_sales)


if __name__ == "__main__":
    wrapper()
    show()
