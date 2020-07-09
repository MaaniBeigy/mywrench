import pandas as pd
import re
def clear_stop_char(text):
    if isinstance(text, pd.core.series.Series):
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
        clear_text = text.replace(stop_dict, regex=True)
        return(clear_text.rstrip())
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
