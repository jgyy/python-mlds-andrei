"""
loops
"""


def wrapper():
    """
    wrapper function
    """
    my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    counter = 0
    for item in my_list:
        counter = counter + item
    print(counter)


if __name__ == "__main__":
    wrapper()
