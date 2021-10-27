"""
NumPy Practice
"""
from numpy import (
    max as npmax,
    min as npmin,
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
    argmax,
    argmin,
    linspace,
    square,
)
from numpy.random import randint, random, seed, randn
from pandas import DataFrame


def wrapper():
    """
    wrapper function
    """
    ar1 = array([1, 2, 3])
    ar2 = array([[1, 2, 3], [4, 5, 6]])
    ar3 = array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]
    )
    print(ar1.shape, ar1.ndim, ar1.dtype, ar1.size, type(ar1))
    print(ar2.shape, ar2.ndim, ar2.dtype, ar2.size, type(ar2))
    print(ar3.shape, ar3.ndim, ar3.dtype, ar3.size, type(ar3))
    daf = DataFrame(ar2)
    print(daf)
    one = ones((10, 2))
    print(one)
    zero = zeros((7, 2, 3))
    print(zero)
    range_array = arange(0, 100, 3)
    print(range_array)
    random_array = randint(10, size=(7, 2))
    print(random_array)
    print(random((3, 5)))
    seed(42)
    print(randint(10, size=(4, 6)))
    arrays = randint(1, 10, size=(3, 7))
    print(unique(arrays))
    print(arrays[0])
    print(arrays[:2])
    print(arrays[:2, :2])
    ar4 = randint(10, size=(3, 5))
    one = ones((3, 5))
    print(ar4 + one)
    print(ar4 - one)
    print(ar4 * one)
    print(ar4 ** 2)
    print(square(ar4))
    print(mean(ar4))
    print(npmax(ar4))
    print(npmin(ar4))
    print(std(ar4))
    print(var(ar4))
    print(ar4.reshape((3, 5, 1)))
    print(ar4.T)

    matrix()


def matrix():
    """
    Create two arrays of random integers between 0 to 10
    one of size (3, 3) the other of size (3, 2)
    """
    mat1 = randint(10, size=(3, 3))
    mat2 = randint(10, size=(3, 2))
    print(dot(mat1, mat2))
    mat3 = randint(10, size=(4, 3))
    mat4 = randint(10, size=(4, 3))
    print(dot(mat3.T, mat4))
    mat5 = randint(10, size=(4, 2))
    mat6 = randint(10, size=(4, 2))
    print(mat5 > mat6)
    print(mat5 >= mat6)
    print(mat5 > 7)
    print(mat5 == mat6)
    print(sort(mat5))
    print(argsort(mat6))
    print(argmax(mat5))
    print(argmin(mat6))
    print(argmax(mat5, axis=1))
    print(argmin(mat6, axis=0))
    print(randn(3, 5))
    print(linspace(1, 100, 10))


if __name__ == "__main__":
    wrapper()
