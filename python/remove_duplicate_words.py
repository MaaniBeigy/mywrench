"""Remove Duplicate Words."""
# ------------------------ Import libraries and functions ---------------------
# import re
# ---------------------------- function definition ----------------------------


def remove_duplicate_words(text: str) -> str:
    """Remove duplicate words.

    It is a general-purpose function, which can remove duplicate words next
    to each other, and preserve only one of them.

    Args:
        text (str): Accepts only one element (i.e., scalar).

    Returns:
        A text variable of <class 'str'> after removing duplicate words.

    Examples:
        >>> input_text = 'Potter Potter I have a message for you'
        >>> remove_duplicate_words(input_text)
        'Potter I have a message for you'

    """
    result = []
    for word in text.split():
        if word not in result:
            result.append(word)
    return " ".join(result)
