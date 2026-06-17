import pandas as pd
from tqdm import tqdm
import random
# import csv_processing
import os
import sys
import json
from dateutil.relativedelta import relativedelta
import torch
import numpy as np
from sentence_transformers.util import cos_sim

import utils.csv_processing as csv_processing

sys.path.append('/home/duy/ly/novelty_checker')


def check_status(df_dataset_,flg_valid_paperId=True,flg_valid_abstract=True,flg_valid_publicationDate=True,flg_valid_hypothesis_4omini=True):

    df_dataset_.fillna('',inplace=True)
    mask_gpt4omini = (df_dataset_['hypothesis_4omini'].str.len() >= 20) & (~df_dataset_['hypothesis_4omini'].str.contains('No result'))
    mask_abstract = (df_dataset_['abstract'].str.len() >= 150) & (~df_dataset_['abstract'].str.contains('No result'))
    mask_publicationDate = (df_dataset_['publicationDate'].str.len() >= 10) & (~df_dataset_['publicationDate'].str.contains('No result'))#len('2025-02-01') = 10 
    mask_paperId = (df_dataset_['paperId'].str.len() >= 20) & (~df_dataset_['paperId'].str.contains('No result'))

    if not flg_valid_paperId:
        mask_paperId = ~mask_paperId

    if not flg_valid_abstract:
        mask_abstract = ~mask_abstract

    if not flg_valid_publicationDate:
        mask_publicationDate = ~mask_publicationDate

    if not flg_valid_hypothesis_4omini:
        mask_gpt4omini = ~mask_gpt4omini

    filtered_df = df_dataset_[mask_paperId & mask_abstract & mask_publicationDate & mask_gpt4omini]
    # filtered_df = filtered_df[~filtered_df['hypothesis_4omini'].str.contains('No result')]
    print(f'\npaperId: {flg_valid_paperId} | abstract: {flg_valid_abstract} | publicationDate: {flg_valid_publicationDate} | hypothesis_4omini: {flg_valid_hypothesis_4omini}: ')
    print(f'{filtered_df.shape[0]} / {df_dataset_.shape[0]}')
    return filtered_df


def select_min_f1_score(df_dataset_,k=1):
    df_negative_clean = pd.DataFrame(columns=df_dataset_.columns)
    # Create empty list to store selected rows
    selected_rows = []
    unique_paperId = df_dataset_['paperId'].unique()

    # Add tqdm progress bar
    for paperId in tqdm(unique_paperId, desc="Processing papers", unit="paper"):
        df_temp = df_dataset_[df_dataset_['paperId'] == paperId]
        if df_temp.shape[0] > 1:
            # Sort by bert_score_f1 in ascending order (to get minimum)
            df_temp = df_temp.sort_values(by='bert_score_f1', ascending=True)
            selected_rows.append(df_temp.iloc[0])  # Take the row with minimum f1 score
        else:
            selected_rows.append(df_temp.iloc[0])

    # Create final DataFrame from the list of selected rows
    df_negative_clean = pd.DataFrame(selected_rows)

    print(f"\nOriginal shape: {df_dataset_.shape}")
    print(f"Clean shape: {df_negative_clean.shape}")
    # Verify the results
    print("\nSample verification:")
    print(df_negative_clean[['paperId', 'bert_score_f1']].head())
    return df_negative_clean


def temporal_train_val_test_split_list(data:list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split data into train, validation, and test sets while preserving temporal order
    Ratios: train:validation:test = 8:1:1
    """
    random.shuffle(data)
    num_samples = len(data)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    train_data = data[:num_train]
    val_data = data[num_train:num_train+num_val]
    test_data = data[num_train+num_val:]
    
    return train_data, val_data, test_data


def temporal_train_val_test_split_df(df_corpus:pd.DataFrame, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split data into train, validation, and test sets while preserving temporal order
    Ratios: train:validation:test = 8:1:1
    """
    # Sort papers by date
    # df_sorted = df_corpus.copy()
    # df_sorted['publicationDate'] = pd.to_datetime(df_sorted['publicationDate'])
    # df_sorted = df_sorted.sort_values('publicationDate')
    df_shuffled = df_corpus.sample(frac=1).reset_index(drop=True)
    # Calculate split indices
    n_samples = len(df_shuffled)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Split data
    train_df = df_shuffled.iloc[:train_size]
    val_df = df_shuffled.iloc[train_size:train_size + val_size]
    test_df = df_shuffled.iloc[train_size + val_size:]
    
    return train_df, val_df, test_df


def load_raw_data(server: str, domain: str, version: str, venue: str = None,split_ratio:str='622') -> dict:
    """
    Load raw data from the server based on the specified domain and venue.

    Args:
        server (str): The server identifier.
        domain (str): The domain of the dataset ('marketing' or 'nlp_papers').
        version (str): The version identifier for the data.
        venue (str, optional): The venue (e.g., 'emnlp', 'acl') if applicable. Defaults to None.

    Returns:
        dict: A dictionary containing the loaded DataFrames:
            - 'df_seed'
            - 'df_ref_all'
            - 'df_seed_negative_subset'
            - 'df_seed_negative_paraphrase'
            - 'df_seed_negative_combination_expanded'
    """
    # Construct the root path based on whether a venue is provided
    if venue is None:
        path_to_root = f'/data/{server}/ly/{domain}/versions/{version}'
    else:
        path_to_root = f'/data/{server}/ly/{domain}/versions/{version}/{venue}'
    
    print(f'path_to_root: {path_to_root}')

    # if domain == 'marketing':
    # path_to_dataset = f'{path_to_root}/{domain}_all.csv'
    if domain == 'nlp_papers':
        path_to_dataset = f'{path_to_root}/{venue}_ref.csv'
        path_to_seed = f'{path_to_root}/{venue}_seed.csv'
    else:
        path_to_dataset = f'{path_to_root}/{domain}_ref.csv'
        path_to_seed = f'{path_to_root}/{domain}_seed.csv'

    df_seed = check_status(csv_processing.read_csv_using_pandas(path_to_seed))
    df_ref = check_status(csv_processing.read_csv_using_pandas(path_to_dataset))

    path_to_seed_negative_combination = f'{path_to_root}/seed_negative_combination.csv'
    path_to_seed_negative_combination_expanded = f'{path_to_root}/seed_negative_combination_expanded.csv'
    path_to_seed_negative_subset = f'{path_to_root}/seed_negative_subset.csv'
    path_to_seed_negative_paraphrase = f'{path_to_root}/seed_negative_paraphrase.csv'
    df_seed_negative_subset = csv_processing.read_csv_using_pandas(path_to_seed_negative_subset)
    df_seed_negative_paraphrase = csv_processing.read_csv_using_pandas(path_to_seed_negative_paraphrase)
    df_seed_negative_combination_expanded = csv_processing.read_csv_using_pandas(path_to_seed_negative_combination_expanded)
    


    path_to_train = f'{path_to_root}/seed_train_{split_ratio}.csv'
    path_to_val = f'{path_to_root}/seed_val_{split_ratio}.csv'
    path_to_test = f'{path_to_root}/seed_test_{split_ratio}.csv'
    path_to_seed_not_used = f'{path_to_root}/seed_not_used.csv'
    df_seed_train = pd.read_csv(path_to_train)
    df_seed_val = pd.read_csv(path_to_val)
    df_seed_test = pd.read_csv(path_to_test)
    df_seed_not_used = pd.read_csv(path_to_seed_not_used)

    # Debug logging of key file paths
    print(f'path_to_seed: {path_to_seed}')
    print(f'path_to_dataset: {path_to_dataset}')
    print(f'path_to_seed_negative_paraphrase: {path_to_seed_negative_paraphrase}')
    print(f'path_to_seed_negative_subset: {path_to_seed_negative_subset}')
    print(f'path_to_seed_negative_combination: {path_to_seed_negative_combination}')
    print(f'path_to_seed_negative_combination_expanded: {path_to_seed_negative_combination_expanded}')

    # Return the collected data in a dictionary
    return {
        'seeds': df_seed,
        'ref': df_ref,
        'seed_negative_subset': df_seed_negative_subset,
        'seed_negative_paraphrase': df_seed_negative_paraphrase,
        'seed_negative_combination_expanded': df_seed_negative_combination_expanded,
        
        'splits': {
            'train': df_seed_train,
            'val': df_seed_val,
            'test': df_seed_test,
            'not_used': df_seed_not_used
        },
        'paths': {
            'path_to_root': path_to_root,
            'path_to_seed': path_to_seed,
            'path_to_ref': path_to_dataset,
            'path_to_seed_negative_paraphrase': path_to_seed_negative_paraphrase,
            'path_to_seed_negative_subset': path_to_seed_negative_subset,
            'path_to_seed_negative_combination': path_to_seed_negative_combination,
            'path_to_seed_negative_combination_expanded': path_to_seed_negative_combination_expanded,
            'path_to_train': path_to_train,
            'path_to_val': path_to_val,
            'path_to_test': path_to_test,
            'path_to_seed_not_used': path_to_seed_not_used
        }
    }


def check_overlap(train_data, val_data, test_data, not_used_data, verbose=True):
    """
    Check for overlaps between train, validation, and test datasets based on the 'hypothesis_4omini' field.

    Args:
        train_data (list of dict): Training dataset.
        val_data (list of dict): Validation dataset.
        test_data (list of dict): Test dataset.
        verbose (bool): Whether to print the overlap stats.

    Returns:
        dict: A dictionary containing overlapping sets:
            - 'intersection_train_val'
            - 'intersection_train_test'
            - 'intersection_val_test'
    """
    train_set = set(train_data['hypothesis_4omini'].tolist())
    val_set = set(val_data['hypothesis_4omini'].tolist())
    test_set = set(test_data['hypothesis_4omini'].tolist())
    not_used_set = set(not_used_data['hypothesis_4omini'].tolist())

    if verbose:
        print(f'Train size      : {len(train_set)}')
        print(f'Validation size : {len(val_set)}')
        print(f'Test size       : {len(test_set)}')
        print(f'Not used size   : {len(not_used_set)}')
        total_size = len(train_set) + len(val_set) + len(test_set) + len(not_used_set)
        print(f'Total size      : {total_size}')
        print('-' * 50)

    intersection_train_val = train_set & val_set
    intersection_train_test = train_set & test_set
    intersection_val_test = val_set & test_set
    intersection_not_used = not_used_set & (train_set | val_set | test_set)

    if verbose:
        print(f'Train ∩ Val  : {len(intersection_train_val)}')
        print(f'Train ∩ Test : {len(intersection_train_test)}')
        print(f'Val ∩ Test   : {len(intersection_val_test)}')
        print(f'Not used ∩ (Train ∪ Val ∪ Test) : {len(intersection_not_used)}')

    return {
        'intersection_train_val': intersection_train_val,
        'intersection_train_test': intersection_train_test,
        'intersection_val_test': intersection_val_test,
        'intersection_not_used': intersection_not_used
    }


def add_corpus(processed_data, df_ref_all, splits=('test',), sample_run=None,target_key='hypothesis_4omini'):
    """
    Adds a 'corpus' to each data point: all reference hypotheses published before the given sample's publication date.

    Args:
        processed_data (dict): A dict of processed samples per split (e.g., from `process_negative_samples()`).
        df_ref_all (pd.DataFrame): Full reference DataFrame with 'hypothesis_4omini' and 'publicationDate'.
        splits (tuple of str): Dataset splits to process (e.g., ('test', 'val')).
        sample_run (int, optional): If set, limits the number of rows processed per split.

    Returns:
        dict: New dictionary with updated entries containing a 'corpus' field.
    """
    df_ref_all.drop_duplicates(subset=['hypothesis_4omini'], inplace=True)
    df_ref_all['publicationDate'] = pd.to_datetime(df_ref_all['publicationDate'], format='mixed', errors='coerce')

    updated_data = {}

    for split in splits:
        data_ = processed_data[split]
        data_df = pd.DataFrame(data_)
        data_df['publicationDate'] = pd.to_datetime(data_df['publicationDate'], format='mixed', errors='coerce')
        data_df['corpus'] = None

        for j, (_, row) in tqdm(enumerate(data_df.iterrows()), total=len(data_df), desc=f'Adding corpus to {split}...'):
            # Get all references published up to that point
            corpus = df_ref_all[
                df_ref_all['publicationDate'] <= row['publicationDate']
            ][target_key].unique().tolist()

            # If the sample is labeled as novel, exclude its own hypothesis from the corpus
            if row['novelty'] == 'Y':
                # print(len(corpus))
                corpus = [x for x in corpus if x != row['given_idea']]
            if row['given_idea'] in corpus:
                corpus.remove(row['given_idea'])
            # elif row['novelty'] =='N':
            for o in row['original']:
                if o not in corpus:
                    corpus.append(o)
            data_df.at[j, 'corpus'] = corpus

            # Optional debug limit
            if sample_run is not None and j >= sample_run:
                break

        data_df['publicationDate'] = data_df['publicationDate'].dt.strftime('%Y-%m-%d')
        updated_data[split] = data_df.to_dict(orient='records')

    return updated_data

def get_neg_type_stats(data, neg_types=None):
    """
    Calculate count and ratio of each neg_type in the provided dataset.

    Args:
        data (list of dict): A list of data points where each has a 'neg_type' key.
        neg_types (list of str, optional): List of negative types to track. If None, it will infer from data.

    Returns:
        tuple: (count_dict, ratio_dict)
            - count_dict: Dictionary of counts for each neg_type.
            - ratio_dict: Dictionary of ratios for each neg_type.
    """
    if neg_types is None:
        neg_types = set(item['neg_type'] for item in data)

    count_dict = {
        neg_type: sum(item['neg_type'] == neg_type for item in data)
        for neg_type in neg_types
    }

    total = len(data)
    ratio_dict = {k: v / total for k, v in count_dict.items()}

    print(f'Total: {total}')
    print(f'Ratio: {ratio_dict}')
    count_dict['true_label'] = sum(item['label'] == 1.0 for item in data)
    count_dict['false_label'] = sum(item['label'] == 0.0 for item in data)
    count_dict['novel'] = sum(item['novelty'] == 'Y' for item in data)
    count_dict['non_novel'] = sum(item['novelty'] == 'N' for item in data)
    print(f'Count: {count_dict}')

    return count_dict, ratio_dict

def shift_negatives_pub_date(target_df, ref_df, months=3, sample_run=None, verbose=False):
    """
    Shift the publication date of each negative sample N months after the corresponding reference's publication date.

    Args:
        target_df (pd.DataFrame): DataFrame containing negative samples. Must have 'hypothesis_4omini' and 'publicationDate'.
        ref_df (pd.DataFrame): Reference DataFrame with original 'hypothesis_4omini' and 'publicationDate'.
        months (int): Number of months to shift the date forward. Default is 3.
        sample_run (int, optional): Limit the number of unique hypotheses to process (for testing/debug).
        verbose (bool): Whether to print detailed logs.

    Returns:
        pd.DataFrame: A copy of target_df with updated 'publicationDate' values.
    """
    updated_df = target_df.copy()
    updated_df['publicationDate'] = pd.to_datetime(updated_df['publicationDate'])
    ref_df['publicationDate'] = pd.to_datetime(ref_df['publicationDate'])

    unique_originals = updated_df['hypothesis_4omini'].unique()

    if verbose:
        print(f'Total unique originals to process: {len(unique_originals)}')

    for i, original in enumerate(unique_originals):
        if sample_run is not None and i >= sample_run:
            break

        same_idx = updated_df[updated_df['hypothesis_4omini'] == original].index
        ref_dates = ref_df[ref_df['hypothesis_4omini'] == original]['publicationDate'].unique().tolist()

        if not ref_dates:
            if verbose:
                print(f'[Warning] No reference date found for: {original}')
            continue

        new_date = pd.to_datetime(ref_dates[0]) + relativedelta(months=months)
        updated_df.loc[same_idx, 'publicationDate'] = new_date

        if verbose:
            print(f'{i}: Updated {len(same_idx)} rows of "{original}" to {new_date}')

    return updated_df
    # updated_subset_df = shift_negatives_pub_date(
    # target_df=data['df_seed_negative_subset'],
    # ref_df=data['df_ref_all'],
    # months=3,
    # sample_run=5,
    # verbose=True)

def build_nc_positive_test_samples(seed_test, df_not_used, ref=None, format_date=True, verbose=False):
    """
    Create positive test samples from a list of seed hypotheses.

    Args:
        seed_test (list): List of novel hypotheses (strings).
        ref (pd.DataFrame): DataFrame with reference hypotheses and their publication dates.
        df_not_used (pd.DataFrame, optional): Extra DataFrame to merge into ref (e.g. unseen papers).
        format_date (bool): Whether to format 'publicationDate' as string (YYYY-MM-DD).
        verbose (bool): Print details for each processed sample.

    Returns:
        list of dict: Positive test samples with required metadata.
    """
    if seed_test is None:
        seed_test = list(set(df_not_used['hypothesis_4omini'].tolist()))
    else:
        seed_test = list(set(seed_test)) + list(set(df_not_used['hypothesis_4omini'].tolist()))

    # ref['publicationDate'] = pd.to_datetime(ref['publicationDate'])

    positive_test = []

    for i, hypothesis in tqdm(enumerate(seed_test), total=len(seed_test), desc='Building positive test set...'):
        match = df_not_used[df_not_used['hypothesis_4omini'] == hypothesis]

        if match.empty:
            if verbose:
                print(f"[Warning] No match found for hypothesis: {hypothesis}")
            continue


        # pub_date = pd.to_datetime(match['publicationDate'].values[0])
        # corpus = ref[ref['publicationDate'] <= pub_date]['hypothesis_4omini'].tolist()
        # corpus = [h for h in corpus if h != hypothesis]
        # pub_date = pd.Timestamp(pub_date).strftime('%Y-%m-%d')
        new_record = {
            'id': match['paperId'].values[0],
            'given_idea': hypothesis,
            'original': ['NA'],
            'neg_type': 'NA',
            'novelty': 'Y',
            'publicationDate': match['publicationDate'].values[0],
            'label': 1.0,
            # 'corpus': corpus+ ['NA']  # Uncomment if corpus is needed later
        }

        if verbose:
            print(f"Added: {new_record}")

        positive_test.append(new_record)

    return positive_test

def prepare_all_data(data,splits=('train', 'val', 'test')):
    processed_ = create_alignment_samples_gpt(data, splits=splits, neg_types=None)

    seed_test =[]
    # seed_test= set([seed_test.extend(x['original']) for x in processed_['test'] if x['label'] == 1.0] + data['splits']['not_used']['hypothesis_4omini'].tolist())
    seed_test= set([seed_test.extend(x['original']) for x in processed_['test'] if x['label'] == 1.0] + data['splits']['not_used']['hypothesis_4omini'].tolist() + data['splits']['train']['hypothesis_4omini'].tolist() + data['splits']['val']['hypothesis_4omini'].tolist())
    processed_['nc'] = [x for x in processed_['test'] if x['label'] == 1.0]
    processed_['nc']  += build_nc_positive_test_samples(seed_test=None,df_not_used=data['splits']['not_used'],ref=None,format_date=False,verbose=False)
    return processed_

def load_preprocseed_data(path_to_data_root,ref,splits=['nc'],if_add_corpus=True):
    processed={}
    for key in splits:
        if key == 'nc':
            path_to_file = f'{path_to_data_root}/{key}_data_all_before_0518.json'
        else:
            path_to_file = f'{path_to_data_root}/retrieval_{key}.json'
        if os.path.isfile(path_to_file):
            with open(path_to_file,'r') as f:
                processed[key] = json.load(f)
    print(processed.keys())
    if if_add_corpus:
        processed = add_corpus(processed,ref,splits=('nc',))

    return processed

def prepare_retrieval_results_for_nc(
    test_data,
    query_emb_dict,
    candidate_emb_dict,
    model_name="vanilla",
    evaluation_type="default",
    top_k=10,
    save_path=None,
    verbose=False
):
    """
    Evaluate retrieval performance and save detailed top-K results.

    Args:
        test_data (list): List of data dicts with keys 'given_idea', 'original', 'corpus', etc.
        query_emb_dict (dict): Dictionary mapping queries to their embeddings.
        candidate_emb_dict (dict): Dictionary mapping candidates to their embeddings.
        model_name (str): Identifier for the model (used in logging/saving).
        evaluation_type (str): Custom string to label output results.
        top_k (int): Number of top candidates to consider.
        save_path (str): Path to save JSON log. If None, auto-generates a default.
        verbose (bool): Whether to print warnings if gold is not in the candidate set.

    Returns:
        dict: Summary metrics (hits@K, MRR, MAP).
    """
    print("🚀 Step 3: Running retrieval per query...")
    hits_at_k = [0] * top_k
    reciprocal_ranks = []
    average_precisions = []
    retrieval_log = []

    for item in tqdm(test_data, desc=f"Evaluating ({model_name})"):
        query = item["given_idea"]
        gold = item["original"]
        candidates = item["corpus"]
        novelty = item["novelty"]

        # if gold not in candidates:
        #     if verbose:
        #         print(f"[WARN] Gold not in candidates for query: {query}")
        #     continue

        try:
            query_embedding = query_emb_dict[query]
            candidate_embeddings = torch.stack([candidate_emb_dict[c] for c in candidates])
        except KeyError as e:
            if verbose:
                print(f"[ERROR] Missing embedding for: {e}")
            continue

        scores = cos_sim(query_embedding, candidate_embeddings)[0]
        top_results = torch.topk(scores, k=min(top_k, len(candidates)))
        date = pd.Timestamp(item['publicationDate']).strftime('%Y-%m-%d')
        result_item = {
            "given_idea": query,
            "original": gold,
            "neg_type": item.get("neg_type", "NA"),
            "publicationDate": date,
            "novelty": novelty,
            "top_k": []
        }

        found = False
        precisions = []

        for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices), start=1):
            retrieved = candidates[idx]
            is_correct = int(retrieved in gold)

            result_item["top_k"].append({
                "rank": rank,
                "sentence": retrieved,
                "score": float(score),
                "is_correct": is_correct
            })

            if rank <= top_k:
                hits_at_k[rank - 1] += is_correct

            if is_correct and not found:
                reciprocal_ranks.append(1.0 / rank)
                found = True

            if is_correct:
                precisions.append(1.0 / rank)

        if not found:
            reciprocal_ranks.append(0.0)

        average_precisions.append(np.mean(precisions) if precisions else 0.0)
        retrieval_log.append(result_item)

    # Save retrieval logs
    if save_path is None:
        save_path = f"./retrieval_results_{model_name}_{evaluation_type}.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_log, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved detailed retrieval results to {save_path}")

    # Compute and print metrics
    cumulative_hits = np.cumsum(hits_at_k)
    num_queries = len(test_data)

    results = {}
    for k in [1, 3, 5, 10]:
        if k <= top_k:
            acc_k = cumulative_hits[k - 1] / num_queries
            print(f"✅ Accuracy@{k}: {acc_k}")
            results[f"Accuracy@{k}"] = acc_k

    mrr = np.mean(reciprocal_ranks)
    map_score = np.mean(average_precisions)
    print(f"📈 MRR@{top_k}: {mrr}")
    print(f"📊 MAP@{top_k}: {map_score}")

    results.update({
        "MRR": mrr,
        "MAP": map_score
    })
    
    return retrieval_log,results


def create_alignment_samples_gpt(data_, splits=('train', 'val', 'test'), neg_types=None,comb_mode=2):

    if neg_types is None:
        neg_types = [
            'seed_negative_paraphrase',
            'seed_negative_subset',
            'seed_negative_combination_expanded'
        ]

    processed = {split: [] for split in splits}
    all_originals_per_split = {
        split: set(data_['splits'][split]['hypothesis_4omini'].unique())
        for split in splits
    }

    # 构建 combination 句与其两个 hypothesis 的映射
    combo_hyp_map = {}
    if 'seed_negative_combination_expanded' in neg_types:
        df_combo = data_['seed_negative_combination_expanded']
        combo_hyp_map = df_combo.groupby('sentence')['hypothesis_4omini'].apply(list).to_dict()
        # print(combo_hyp_map)

    def get_hypothesis_split(hyp):
        for s in splits:
            if hyp in all_originals_per_split[s]:
                return s
        return None

    def determine_combination_split(h1, h2):
        s1, s2 = get_hypothesis_split(h1), get_hypothesis_split(h2)
        if s1 == s2:
            return s1
        if s1 is None and s2 is not None:
            return s2
        if s1 is not None and s2 is None:
            return s1
        if s1 is None and s2 is None:
            return None

        pair = {s1, s2}

        if pair == {'train', 'val'}:
            return 'train'
        elif pair == {'val', 'test'}:
            return 'val'
        elif pair == {'train', 'test'}:
            return 'train'
        else:
            print(f"⚠️ Unexpected split pair: ({s1}, {s2}) for ({h1}, {h2})")
            return None

    for i, split in enumerate(splits):
        split_df = data_['splits'][split]
        print(f'{i + 1}/{len(splits)} | {split}: {len(split_df)} samples...')
        originals = split_df['hypothesis_4omini'].unique().tolist()

        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f'Processing {split}...'):
            for neg_type in neg_types:
                df_neg = data_[neg_type]
                match_rows = df_neg[df_neg['hypothesis_4omini'] == row['hypothesis_4omini']]

                for _, neg_row in match_rows.iterrows():
                    sent = neg_row['sentence']
                    ntype = neg_row['neg_type']
                    pub_date = neg_row['publicationDate']

                    # ✅ 处理组合句（仅允许 test 中出现 list original）
                    if neg_type == 'seed_negative_combination_expanded':
                        hyps = list(set(combo_hyp_map.get(sent, [])))
                        if len(hyps) != 2:
                            # print(f'hyps: {hyps}')
                            # print(f"⚠️ Combination sentence without exactly 2 hypotheses: {len(hyps)}: {sent}")
                            # break
                            continue

                        hyp1, hyp2 = hyps
                        assigned_split = determine_combination_split(hyp1, hyp2) or split

                        if assigned_split == 'test':
                            if comb_mode == 1:
                                processed['test'].append({
                                    'id': neg_row['paperId'],
                                    'given_idea': sent,
                                    'original': [hyp1, hyp2],
                                    'neg_type': ntype,
                                    'novelty': 'N',
                                    'publicationDate': pub_date,
                                    'label': 1.0
                                })
                            elif comb_mode == 2:
                                for hyp in [hyp1, hyp2]:
                                    processed['test'].append({
                                        'id': neg_row['paperId'],
                                        'given_idea': sent,
                                        'original': [hyp],
                                        'neg_type': ntype,
                                        'novelty': 'N',
                                        'publicationDate': pub_date,
                                        'label': 1.0
                                    })
                        else:
                            # ⛔ 若组合句分配到 train/val，仅创建两个单独样本
                            for hyp in [hyp1, hyp2]:
                                processed[assigned_split].append({
                                    'id': neg_row['paperId'],
                                    'given_idea': sent,
                                    'original': hyp,
                                    'neg_type': ntype,
                                    'novelty': 'N',
                                    'publicationDate': pub_date,
                                    'label': 1.0
                                })
                                # val 增加一个错误配对样本
                                if assigned_split == 'val':
                                    pool = list(set(originals) - {hyp})
                                    if pool:
                                        rand_hyp = random.choice(pool)
                                        processed['val'].append({
                                            'id': neg_row['paperId'],
                                            'given_idea': sent,
                                            'original': rand_hyp,
                                            'neg_type': ntype,
                                            'novelty': 'N',
                                            'publicationDate': pub_date,
                                            'label': 0.0
                                        })

                    else:

                        if split == 'test':
                            processed[split].append({
                            'id': neg_row['paperId'],
                            'given_idea': sent,
                            'original': [neg_row['hypothesis_4omini']],
                            'neg_type': ntype,
                            'novelty': 'N',
                            'publicationDate': pub_date,
                            'label': 1.0
                        })
                        else:

                            # ✅ 非组合负例
                            processed[split].append({
                                'id': neg_row['paperId'],
                                'given_idea': sent,
                                'original': neg_row['hypothesis_4omini'],
                                'neg_type': ntype,
                                'novelty': 'N',
                                'publicationDate': pub_date,
                                'label': 1.0
                            })

                            # val 增加一个错误配对样本
                            if split == 'val':
                                pool = list(set(originals) - {neg_row['hypothesis_4omini']})
                                if pool:
                                    rand_hyp = random.choice(pool)
                                    processed[split].append({
                                        'id': neg_row['paperId'],
                                        'given_idea': sent,
                                        'original': rand_hyp,
                                        'neg_type': ntype,
                                        'novelty': 'N',
                                        'publicationDate': pub_date,
                                        'label': 0.0
                                    })

        # # 可选统计输出（需你本地定义 dataset.get_neg_type_stats）
        # if 'dataset' in globals():
        count_dict, ratio_dict = get_neg_type_stats(processed[split])
        print()

    return processed

def create_alignmnet_samples_using_ref(data_, splits=('train', 'val', 'test'), neg_types=None):

    """
    Processes negative samples by matching paper IDs from reference data and combining related negatives.

    Args:
        data (dict): Dictionary containing:
            - 'ref': reference DataFrame (must include 'hypothesis_4omini')
            - 'splits': dict of split DataFrames (e.g., {'train': df_train, 'val': df_val, ...})
            - Each neg_type DataFrame with 'paperId', 'sentence', 'hypothesis_4omini', 'neg_type', 'publicationDate'
        splits (tuple): Which dataset splits to process.
        neg_types (list): List of negative DataFrame keys in `data` to pull from.

    Returns:
        dict: A dictionary with processed negative samples for each split.
    """
    processed = {}
    # data_ = data.copy()ffdd
    for i, split in enumerate(splits):
        
        split_df = data_['splits'][split]
        print(f'{split}: {len(split_df)}')
        processed[split] = []
        # print(f'{i + 1}/{len(splits)} | {split}: {len(split_df)} samples...')
        originals = split_df['hypothesis_4omini'].unique().tolist()
        # print(len(originals))
        all_ref = set(data_['ref']['hypothesis_4omini'].unique().tolist())
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f'Processing {split}...'):
            paperId = row['paperId']
            refs = data_['ref'][ data_['ref']['cited_by']==paperId]['hypothesis_4omini'].unique().tolist()
            if split != 'test':
                # print(f"Adding {len(refs)} positives for {row['paperId']}")
                for ref in refs:
                    processed[split].append({
                        'id': row['paperId'],
                        'given_idea': row['hypothesis_4omini'],
                        'original': ref,
                        'neg_type': 'NA',
                        'novelty': 'Y',
                        'publicationDate': row['publicationDate'],
                        'label': 1.0
                    })
            if split ==  'val':
                fake_refs = random.sample(list(all_ref-set(refs)),len(refs))
                for fake_ref in fake_refs:
                    processed[split].append({
                        'id': row['paperId'],
                        'given_idea': row['hypothesis_4omini'],
                        'original': fake_ref,
                        'neg_type': 'NA',
                        'novelty': 'Y',
                        'publicationDate': row['publicationDate'],
                        'label': 0.0
                    })

        # count_dict, ratio_dict = get_neg_type_stats(processed[split])
    return processed


def check_neg_type_status(test_data:list):
    neg_types = ['paraphrase', 'subset', 'combination','NA']
    print('='*100)
    print(f"Total: {len(test_data)}")
    print('-'*100)
    for neg_type in neg_types:
        temp = [x for x in test_data if x['neg_type'] == neg_type]
        print(f"{neg_type}: {len(temp)}")
        # print('-'*100)


def check_if_all_original_in_corpus(test_data:list):
    for x in test_data:
        if x['original'] != 'NA':
            if not set(x['original']).issubset(set(x['corpus'])) and x['neg_type'] != 'NA':
                print(x['original'])


def check_dataset_overlap(processed_):
    # 🧱 收集 train / val 原始 hypothesis
    train_set = set(x['original'] for x in processed_['train'] if x['original'] != 'NA' and x['label'] == 1.0)
    val_set = set(x['original'] for x in processed_['val'] if x['original'] != 'NA' and x['label'] == 1.0)

    # 🧪 收集 test 中的组合型 original（列表形式）
    test_set = set()
    for item in processed_['test']:
        if isinstance(item['original'], list) and item['original'] != ['NA'] and item['label'] == 1.0:
            test_set.update(item['original'])

    # ✅ 尺寸统计
    print(f"Train size      : {len(train_set)}")
    print(f"Validation size : {len(val_set)}")
    print(f"Test size       : {len(test_set)}")
    print("-" * 50)

    # ❌ 重复交集检查
    intersection_train_val  = train_set & val_set
    intersection_train_test = train_set & test_set
    intersection_val_test   = val_set  & test_set

    print(f"Train ∩ Val  : {len(intersection_train_val)}")
    print(f"Train ∩ Test : {len(intersection_train_test)}")
    print(f"Val ∩ Test   : {len(intersection_val_test)}")

    # ✅ 总量确认
    total_size = len(train_set | val_set | test_set)
    print(f"Total unique original hypotheses: {total_size}")

def load_json(path_to_result):
    try:
        with open(path_to_result, 'r') as f:
            lines = json.load(f)
    except:
        lines = []
        with open(path_to_result, 'r') as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))
    return lines

def check_retrieval_results(logs,model_name):
    analysis = pd.DataFrame(columns=['model_name','data_type','is_correct_retrieval','is_correct_gold','count'])

    for data_type in ['N','Y']:
        # 0 0 n
        temp = [x for x in logs if (x['correct_retrieval_rank'] == 0) and (x['is_correct'] == False) and (x['gold'] == data_type.lower())]
        analysis = pd.concat([analysis,pd.DataFrame({'model_name':[model_name],
                                                     'data_type':[data_type],'is_correct_retrieval':[False],'is_correct_gold':[False],'count':[len(temp)]})])
        # 0 1 n
        temp = [x for x in logs if (x['correct_retrieval_rank'] == 0) and (x['is_correct'] == True) and (x['gold'] == data_type.lower())]          
        analysis = pd.concat([analysis,pd.DataFrame({'model_name':[model_name],
                                                    'data_type':[data_type],'is_correct_retrieval':[False],'is_correct_gold':[True],'count':[len(temp)]})])
        # 1 0 n
        temp = [x for x in logs if (x['correct_retrieval_rank'] > 0) and (x['is_correct'] == False) and(x['gold'] == data_type.lower())]
        analysis = pd.concat([analysis,pd.DataFrame({'model_name':[model_name],
                                                    'data_type':[data_type],'is_correct_retrieval':[True],'is_correct_gold':[False],'count':[len(temp)]})])
        # 1 1 n
        temp = [x for x in logs if (x['correct_retrieval_rank'] > 0) and (x['is_correct'] == True) and (x['gold'] == data_type.lower())]
        analysis = pd.concat([analysis,pd.DataFrame({'model_name':[model_name],
                                                    'data_type':[data_type],'is_correct_retrieval':[True],'is_correct_gold':[True],'count':[len(temp)]})])

    return analysis