"""Cleaning Spaces."""
# ------------------------ Import libraries and functions ---------------------
import re

# ---------------------------- function definition ----------------------------


def space_cleaner(text: str) -> str:
    """Cleans extra spaces.

    Args:
        text (str): the text to be cleaned.

    Returns:
        A text variable of <class 'str'> after separating numbers.

    Examples:
        >>> space_cleaner("hello to  people ")
        'hello to people'

    """
    clear_text = re.sub("  ", " ", text).strip()
    return clear_text
