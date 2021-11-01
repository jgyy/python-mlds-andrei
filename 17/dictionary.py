"""
lists 3
"""


def wrapper():
    """
    wrapper function
    """
    user = {
        "age": 22,
        "username": "Shogun",
        "weapons": ["katana", "shuriken"],
        "is_active": True,
        "clan": "Japan",
    }
    print(user.keys())
    user["weapons"].append("shield")
    print(user)
    user.update({"is_banned": False})
    print(user)
    user["is_banned"] = True
    print(user)
    user2 = user.copy()
    user2.update({"age": 100, "username": "Timbo"})
    print(user2)


if __name__ == "__main__":
    wrapper()
