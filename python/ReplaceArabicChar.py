import pandas as pd
import re
def replace_arabic_char(text):
    arabic_dict = {
            'ي': 'ی',
            'ك': 'ک',
            ' ً' : '',
            ' ٌ' : '',
            ' ٍ' : '',
            ' َ' : '',
            ' ُ' : '',
            ' ِ' : '',
            ' ّ' : '',
            ' ْ' : ''
        }
    if isinstance(text, pd.core.series.Series):
        clear_text = text.replace(arabic_dict, regex=True)
        return(clear_text)
    elif isinstance(text, str):
        rep = dict((re.escape(k), v) for k, v in arabic_dict.items())
        pattern = re.compile("|".join(rep.keys()))
        clear_text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
        return(clear_text)