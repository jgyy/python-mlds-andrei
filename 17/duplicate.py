"""
gui exercise
"""


def wrapper():
    """
    wrapper function
    """
    some_list = ["a", "b", "c", "b", "d", "m", "n", "n"]
    duplicates = []
    for value in some_list:
        if some_list.count(value) > 1:
            if value not in duplicates:
                duplicates.append(value)
    print(duplicates)


if __name__ == "__main__":
    wrapper()
