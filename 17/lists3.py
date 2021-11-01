"""
lists 3
"""


def wrapper():
    """
    wrapper function
    """
    friends = ["Simon", "Patty", "Joy", "Carrie", "Amira", "Chu"]
    new_friend = ["Stanley"]
    friends.sort()
    print(friends + new_friend)
    friends.extend(new_friend)
    print(sorted(friends))


if __name__ == "__main__":
    wrapper()
