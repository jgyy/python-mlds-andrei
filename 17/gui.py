"""
gui exercise
"""


def wrapper():
    """
    wrapper function
    """
    picture = [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ]
    for image in picture:
        for pixel in image:
            if pixel:
                print("*", end="")
            else:
                print(" ", end="")
        print("")


if __name__ == "__main__":
    wrapper()
