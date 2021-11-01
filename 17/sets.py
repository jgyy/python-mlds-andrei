"""
sets
"""


def wrapper():
    """
    wrapper function
    """
    school = {"Bobby", "Tammy", "Jammy", "Sally", "Danny"}
    attendance_list = ["Jammy", "Bobby", "Danny", "Sally"]
    print(school.difference(attendance_list))


if __name__ == "__main__":
    wrapper()
