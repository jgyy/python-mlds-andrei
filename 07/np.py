"""
Introduction to Numpy
"""
from os.path import join, dirname
from timeit import timeit
from numpy import (
    sum as npsum,
    min as npmin,
    max as npmax,
    array,
    ones,
    zeros,
    arange,
    unique,
    mean,
    std,
    var,
    dot,
    sort,
    argsort,
    argmin,
    argmax,
    sqrt,
    square,
    add,
    exp,
    log,
)
from numpy.random import random, randint, rand, seed
from pandas import DataFrame
from matplotlib.pyplot import hist, figure, show
from matplotlib.image import imread

imreads = lambda x: imread(join(dirname(__file__), x))


def wrapper():
    """
    wrapper function
    """
    da1 = array([1, 2, 3])
    print(da1)
    print(type(da1))
    da2 = array([[1, 2.0, 3.3], [4, 5, 6.5]])
    da3 = array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]
    )
    print(da2)
    print(da3)
    print(da1.shape)
    print(da2.shape)
    print(da3.shape)
    print(da1)
    print(da2)
    print(da3)
    print(da1.ndim, da2.ndim, da3.ndim)
    print(da1.dtype, da2.dtype, da3.dtype)
    print(da3)
    print(da1.size, da2.size, da3.size)
    print(type(da1), type(da2), type(da3))
    print(da2)
    daf = DataFrame((da2))
    print(daf)

    create_array(da1, da2, da3)


def create_array(da1, da2, da3):
    """
    Creating arrays function
    """
    sample_array = array([1, 2, 3])
    print(sample_array)
    print(sample_array.dtype)
    one = ones((2, 3))
    print(one)
    print(one.dtype)
    print(type(one))
    zero = zeros((2, 3))
    print(zero)
    range_array = arange(0, 10, 2)
    print(range_array)
    random_array = randint(0, 10, size=(3, 5))
    print(random_array)
    print(random_array.size)
    print(random_array.shape)
    random_array_2 = random((5, 3))
    print(random_array_2)
    print(random_array_2.shape)
    random_array_3 = rand(5, 3)
    print(random_array_3)
    seed(seed=99999)
    random_array_4 = randint(10, size=(5, 3))
    print(random_array_4)
    seed(7)
    random_array_5 = random((5, 3))
    print(random_array_5)
    random_array_5 = random((5, 3))
    print(random_array_5)
    print(random_array_4)
    print(unique(random_array_4))

    array_matrix(da1, da2, da3)


def array_matrix(da1, da2, da3):
    """
    Viewing arrays and matrices function
    """
    print(da1)
    print(da2)
    print(da3)
    print(da1[0])
    print(da2.shape)
    print(da2[0])
    print(da3.shape)
    print(da3[0])
    print(da3)
    print(da2)
    print(da2[1])
    print(da3)
    print(da3.shape)
    print(da3[:2, :2, :2])
    da4 = randint(10, size=(2, 3, 4, 5))
    print(da4)
    print(da4.shape, da4.ndim)
    print(da4[:, :, :, :1])

    manipulate(da1, da2, da3)


def manipulate(da1, da2, da3):
    """
    Manipulating and comparing arrays function
    """
    print(da1)
    one = ones(3)
    print(one)
    print(da1 + one)
    print(da1 - one)
    print(da1 * one)
    print(da1)
    print(da2)
    print(da1 * da2)
    print(da3)
    print(da1 / one)
    print(da2 / da1)
    print(da2 // da1)
    print(da2)
    print(da2 ** 2)
    print(square(da2))
    print(da1 + one)
    print(add(da1, one))
    print(da1 % 2)
    print(da1 / 2)
    print(da2 % 2)
    print(exp(da1))
    print(log(da1))
    listy_list = [1, 2, 3]
    print(type(listy_list))
    print(npsum(listy_list))
    print(npsum(da1))
    massive_array = random(100000)
    print(massive_array.size)
    print(massive_array[:10])
    print(
        timeit(stmt="sum(numpy.random.random(100000))", setup="import numpy", number=1)
    )
    print(
        timeit(
            stmt="numpy.sum(numpy.random.random(100000))",
            setup="import numpy",
            number=1,
        )
    )

    after_time(da1, da2, da3)


def after_time(da1, da2, da3):
    """
    After performing the timeit function to see numpy speed
    """
    print(17900 / 34)
    print(da2)
    print(mean(da2))
    print(npmax(da2))
    print(npmin(da2))
    print(std(da2))
    print(var(da2))
    print(sqrt(var(da2)))
    high_var_array = array([1, 100, 200, 300, 4000, 5000])
    low_var_array = array([2, 4, 6, 8, 10])
    print(var(high_var_array), var(low_var_array))
    print(std(high_var_array), std(low_var_array))
    print(mean(high_var_array), mean(low_var_array))
    figure()
    hist(high_var_array)
    figure()
    hist(low_var_array)
    print(da2)
    print(da2.shape)
    print(da3)
    print(da3.shape)
    print(da2.shape)
    print(da2.reshape(2, 3, 1).shape)
    print(da3.shape)
    a2_reshape = da2.reshape(2, 3, 1)
    print(a2_reshape)
    print(a2_reshape * da3)
    print(da2)
    print(da2.shape)
    print(da2.T)
    print(da2.T.shape)
    print(da3)
    print(da3.shape)
    print(da3.T)
    print(da3.T.shape)

    dot_product(da1, da2)


def dot_product(da1, da2):
    """
    dot product function
    """
    seed(0)
    mat1 = randint(10, size=(5, 3))
    mat2 = randint(10, size=(5, 3))
    print(mat1)
    print(mat2)
    print(mat1.shape, mat2.shape)
    print(mat1)
    print(mat2)
    print(mat1 * mat2)
    print(mat1.T)
    print(mat1.shape, mat2.T.shape)
    mat3 = dot(mat1, mat2.T)
    print(mat3)
    seed(0)
    sales_amounts = randint(20, size=(5, 3))
    print(sales_amounts)
    weekly_sales = DataFrame(
        sales_amounts,
        index=["Mon", "Tues", "Wed", "Thurs", "Fri"],
        columns=["Almond butter", "Peanut butter", "Cashew butter"],
    )
    print(weekly_sales)
    prices = array([10, 8, 12])
    print(prices)
    print(prices.shape)
    butter_prices = DataFrame(
        prices.reshape(1, 3),
        index=["Price"],
        columns=["Almond butter", "Peanut butter", "Cashew butter"],
    )
    print(butter_prices)
    print(prices.shape)
    print(sales_amounts.shape)
    total_sales = prices.dot(sales_amounts.T)
    print(total_sales)
    print(butter_prices.shape, weekly_sales.shape)
    print(weekly_sales.T.shape)
    daily_sales = butter_prices.dot(weekly_sales.T)
    print(daily_sales.shape)
    print(weekly_sales.shape)
    print(weekly_sales)
    weekly_sales["Total ($)"] = daily_sales.T
    print(weekly_sales)

    comparison(da1, da2)


def comparison(da1, da2):
    """
    comparison operators function
    """
    print(da1)
    print(da2)
    print(da1 > da2)
    bool_array = da1 >= da2
    print(bool_array)
    print(type(bool_array), bool_array.dtype)
    print(da1 > 5)
    print(da1 < 5)
    print(da1)
    print(da2)
    print(da1 == da2)
    random_array = randint(10, size=(3, 5))
    print(random_array)
    print(random_array.shape)
    print(sort(random_array))
    print(random_array)
    print(argsort(random_array))
    print(da1)
    print(argsort(da1))
    print(argmin(da1))
    print(argmax(da1))
    print(random_array)
    print(argmax(random_array, axis=0))
    print(argmax(random_array, axis=1))
    panda = imreads("numpy-panda.png")
    print(type(panda))
    print(panda.size, panda.shape, panda.ndim)
    print(panda[:5])
    car = imreads("numpy-car-photo.png")
    print(type(car))
    print(car[:1])
    dog = imreads("numpy-dog-photo.png")
    print(type(dog))
    print(dog)


if __name__ == "__main__":
    wrapper()
    show()
