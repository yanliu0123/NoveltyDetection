import numpy as np
import pandas as pd
import re
import time

def reformat_date(date_str):
    """
    Reformats date from 'DD/M/YYYY' to 'YYYY-MM-DD'
    
    Args:
        date_str (str): Date string in format 'DD/M/YYYY'
        
    Returns:
        str: Reformatted date string in 'YYYY-MM-DD' format
    """
    try:
        # Split the date string
        day, month, year = date_str.split('/')
        
        # Pad month and day with leading zeros if needed
        month = month.zfill(2)
        day = day.zfill(2)
        
        # Return formatted date
        return f'{year}-{month}-{day}'
    except:
        return date_str
    
def apply_reformat_date(series):
    return series.apply(reformat_date)


def compare_lists(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    
    # Elements in list1 but not in list2
    only_in_list1 = list(set1 - set2)
    
    # Elements in list2 but not in list1
    only_in_list2 = list(set2 - set1)
    
    # All different elements (symmetric difference)
    all_differences = list(set1.symmetric_difference(set2))

    print("Only in list1:", only_in_list1)
    print("Only in list2:", only_in_list2)
    print("All differences:", all_differences)
    
    return {
        'only_in_list1': only_in_list1,
        'only_in_list2': only_in_list2,
        'all_differences': all_differences
    }

def time_format(seconds):
    days = int(seconds // 86400)
    hours = int(seconds // 3600 %24)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f'{days}d:{hours}h:{minutes}m:{seconds}s'

def is_valid_hypothesis(h):
    invalid_values = ['-1', '', '[]', 'None', '[None]', np.nan]
    return (h not in invalid_values) and (pd.notna(h)) and (len(h) > 10)  

def is_valid_abstract(abstract):
    return (abstract is not None) & ('No result' not in abstract) & (len(abstract) > 100)

def is_valid_publicationDate(publicationDate):
    return (publicationDate is not None) & ('No result' not in publicationDate) & (len(str(publicationDate)) == 10 )

def is_valid_paperId(paperId):
    return (paperId is not None) & ('No result' not in paperId) & (str(paperId)!='nan')

def is_from_acl_anthology_id(paperId):
    return ('acl' in paperId) or ('emnlp' in paperId) or ('naacl' in paperId) or ('eacl' in paperId) or ('conill' in paperId)

