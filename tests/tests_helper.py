import tests.helpers


def list_has(value, lst):
    found = False
    for val in lst:
        if val == value:
            found = True
            break
    return found
