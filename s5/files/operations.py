from utils import *


def calculate():
    a_add_b = add(1, 5)
    a_sub_b = sub(1, 5)
    a_divide_b = divide(1, 5)
    a_multiply_b = multiply(2, 5)

    return a_add_b, a_sub_b, a_divide_b, a_multiply_b

if __name__ == "__main__":
    print(calculate())