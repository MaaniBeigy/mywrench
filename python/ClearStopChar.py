import pandas as pd
import re
def clear_stop_char(text):
    stop_dict = {
        '\\/': '',
        '\\[': '',
        '\\]': '',
        '\\:': '',
        '\\|': '',
        '\\"': '',
        '\\?': '',
        '\\<': '',
        '\\>': '',
        '\\,': '',
        '\\(': '',
        '\\)': '',
        '\\\\': '',
        '\\.': '',
        '\\+': '',
        '\\-': '',
        '\\!': '',
        '\\$': '',
        '\\`': '',
        '\\_': '',
        '^\\s+': '',  # returns string w/o leading whitespace
        '\\s+$': ''  # returns string w/o trailing whitespace
        }
    if isinstance(text, pd.core.series.Series):
        clear_text = text.replace(stop_dict, regex=True)
        return(clear_text)
    elif isinstance(text, str):
        reps = [
            '\\/',
            '\\[',
            '\\]',
            '\\:',
            '\\|',
            '\\"',
            '\\?',
            '\\<',
            '\\>',
            '\\,',
            '\\(',
            '\\)',
            '\\\\',
            '\\.',
            '\\+',
            '\\-',
            '\\!',
            '\\$',
            '\\`',
            '\\_',
            '^\\s+',  # returns string w/o leading whitespace
            '\\s+$'  # returns string w/o trailing whitespace
        ]
        pattern = '|'.join(reps)
        clear_text = re.sub(pattern, '', text)
        return(clear_text)