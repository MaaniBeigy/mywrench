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
        rep = dict((re.escape(k), v) for k, v in stop_dict.items())
        pattern = re.compile("|".join(rep.keys()))
        clear_text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
        return(clear_text)