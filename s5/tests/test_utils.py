import os
import sys

import pytest

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Now you can perform relative imports
from files.utils import *


def test_add():
    assert 1 + 2 == 3

def test_sub():
    assert sub(2, 1) == 1

def test_divide():
    assert divide(3, 2) == 1.5, "This division should return a 1.5"

# Test that checks if division returns a float number
def test_divide2():
    assert type(divide(3, 2)) is float, "This division should return a float number"

@pytest.mark.skipif(not os.path.exists(os.path.join(parent_dir, "data", "data.txt")), reason="Data files not found")
def test_something_about_data():
    pass

@pytest.mark.parametrize("a, b, expected", [(3, 5, 8), (2, 4, 6), (6, 9, 15)])
def test_add_multiple(a, b, expected):
    assert add(a, b) == expected, "Addition of two numbers failed"
