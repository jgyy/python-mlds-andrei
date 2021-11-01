"""
string formatting
"""


def wrapper():
    """
    wrapper function
    """
    print("Hello {}, your balance is {}.".format("Cindy", 50))
    print("Hello {0}, your balance is {1}.".format("Cindy", 50))
    print("Hello {name}, your balance is {amount}.".format(name="Cindy", amount=50))
    print("Hello {0}, your balance is {amount}.".format("Cindy", amount=50))
    name = "Cindy"
    amount = 50
    print(f"Hello {name}, your balance is {amount}.")


if __name__ == "__main__":
    wrapper()
