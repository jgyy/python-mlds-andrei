"""
matrix
"""


def wrapper():
    """
    wrapper function
    """
    basket = ["Banana", ["Apples", ["Oranges"], "Blueberries"]]
    banana = basket[0]
    apples = basket[1][0]
    oranges = basket[1][1][0]
    blueberries = basket[1][2]
    print(banana, apples, oranges, blueberries)


if __name__ == "__main__":
    wrapper()
