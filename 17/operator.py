"""
Operator Precedence
"""


def wrapper():
    """
    wrapper function
    """
    print((5 + 4) * 10 / 2)
    print(((5 + 4) * 10) / 2)
    print((5 + 4) * (10 / 2))
    print(5 + (4 * 10) / 2)
    print(5 + 4 * 10 // 2)


if __name__ == "__main__":
    wrapper()
