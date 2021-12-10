"""Human-Readable Numbers Format."""
# ------------------------ Import libraries and functions ---------------------
from typing import Union

# ---------------------------- function definition ----------------------------


def human_number(number: Union[int, float]) -> str:
    """human-readable number format.

    It is a general-purpose number formatter, which separates every three
    digits to make it human-readable.

    Args:
        number (int or float): Accepts only one element (i.e., scalar).

    Returns:
        A text variable of <class 'str'> after separating numbers.

    Examples:
        >>> human_number(165485613)
        '165,485,613'

    """
    return f"{number:,}"
