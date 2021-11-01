"""
nonlocal
"""


def wrapper():
    """
    wrapper function
    """

    def outer():
        loc = "local"

        def inner():
            nonlocal loc
            loc = "nonlocal"
            print("inner:", loc)

        inner()
        print("outer:", loc)

    outer()


if __name__ == "__main__":
    wrapper()
