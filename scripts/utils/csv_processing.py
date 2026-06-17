import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer #cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def read_csv_using_pandas(path_to_csv):
    encodings = ['utf-8', 'iso-8859-1', 'cp1252','latin-1', ]
    for encoding in encodings:
        try:
            print(f'Reading {path_to_csv} with encoding = {encoding}')
            df = pd.read_csv(path_to_csv, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Failed to read {path_to_csv} with any encoding")


def print_row(df_dataset,indexes):
    if type(indexes) == int:
        indexes = [indexes]
    for index in indexes:
        print('-'*20, f'{index+2}/{len(df_dataset)}', '-'*20)
        for column in df_dataset.columns:
            print(f'{column} = {df_dataset.loc[index][column]}')
        print()


def update_dataset(df_dataset_, dict_to_update, index_to_update):
    """
    Update DataFrame with values for multiple indices.
    
    Args:
        df_dataset_: DataFrame to update
        dic_to_update: Dictionary of column:value pairs
        index_to_update: List of indices to update
    """
    # Create a temporary DataFrame with the same values repeated
    temp_df = pd.DataFrame(
        {col: [val] * len(index_to_update) for col, val in dict_to_update.items()},
        index=index_to_update
    )
    
    # Update only existing columns
    existing_cols = [col for col in dict_to_update.keys() if col in df_dataset_.columns]
    df_dataset_.loc[index_to_update, existing_cols] = temp_df[existing_cols]
    print(f'Updated {len(index_to_update)} rows: {index_to_update}')
    for x in dict_to_update.keys():
        print(f'\t{x}: {dict_to_update[x]}')
    return df_dataset_


def find_similar_titles(target_title, all_titles, similarity_threshold=0.8):
    """
    Find titles that are similar to the target title using cosine similarity.
    
    Args:
        target_title: Title to compare against
        all_titles: Series/array of titles to check
        similarity_threshold: Minimum similarity score to consider a match
    
    Returns:
        numpy array of boolean masks indicating similar titles
    """
    all_titles = all_titles.fillna('')
    vectorizer = TfidfVectorizer()
    
    if 'list' not in str(type(all_titles)):
    # Create title list including target
        titles_list = [target_title] + all_titles.tolist()
    else:
        titles_list = [target_title] + all_titles
    try:
        # Calculate TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(titles_list)
        
        # Calculate similarity between target and all other titles
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        # Return boolean mask for similar titles
        return similarities >= similarity_threshold
        
    except Exception as e:
        print(f"Error computing similarities for title '{target_title}': {str(e)}")
        return np.zeros(len(all_titles), dtype=bool)
    

def clean_invalid_rows(df_dataset_):
    mask_seed_title = df_dataset_['seed_title'].isna()
    # mask_inline_citation = df_dataset_['inline_citation'].na()
    mask_raw_text =  df_dataset_['raw_text'].str.contains('@') | df_dataset_['raw_text'].str.contains('\*')
    mask_invalid = mask_seed_title & mask_raw_text
    index_to_drop = df_dataset_[mask_invalid].index.tolist()

    return index_to_drop



