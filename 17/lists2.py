"""
lists 2
"""


def wrapper():
    """
    wrapper function
    """
    basket = ["Banana", "Apples", "Oranges", "Blueberries"]
    print(basket)
    basket.remove('Banana')
    print(basket)
    basket.pop()
    print(basket)
    basket.append('Kiwi')
    print(basket)
    basket.insert(0, 'Apples')
    print(basket)
    print(basket.count('Apples'))
    basket.clear()
    print(basket)


if __name__ == "__main__":
    wrapper()
