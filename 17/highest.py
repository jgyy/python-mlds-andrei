"""
highest-even-solution
"""


def wrapper():
    """
    wrapper function
    """

    def highest_even(lis):
        evens = []
        for item in lis:
            if item % 2 == 0:
                evens.append(item)
        return max(evens)

    print(highest_even([10, 1, 2, 3, 4, 8]))


if __name__ == "__main__":
    wrapper()
