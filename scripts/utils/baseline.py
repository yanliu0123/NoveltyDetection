import ast
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from typing import List, Dict, Tuple
import numpy as np
import spacy
from scipy.spatial.distance import cosine
from itertools import combinations
## Tools
def safe_eval_list(val):
    # try:
    
    if val == '' or val is None:
        return []
    # print(f'val: {val}, {type(val)}')
    val = str(val).replace("'s", '')
    try:    
        result = ast.literal_eval(val)
        if isinstance(result, list):
            return list(set(result))
        else:
            result = val.replace('[','').replace(']','').replace('\'','').split(',')
            result = [r.strip() for r in result]
            return list(set(result))
    except Exception as e:
        vals = val.replace('[','').replace(']','').replace('\'','').split(',')
        result = []
        for r in vals:
            if r == '':
                continue
            if r[0] == "'":
                r = r[1:]
            if r[-1] == "'":
                r = r[:-1]
            result.append(r.strip())
        return list(set(result))

def print_metrics(y_true: List[str], y_pred: List[str]) -> None:
    if 1 in y_true or 0 in y_true:
        y_true = ['y' if x == 1 else 'n' for x in y_true]
        y_pred = ['y' if x == 1 else 'n' for x in y_pred]
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label="y")

    print("\n📊 Evaluation Metrics:")
    print(f"Accuracy: {acc:.5f}")
    print(f"Precision: {precision:.5f}, Recall: {recall:.5f}, F1: {f1:.5f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=5))
    
def print_dict_structure(d, indent=0):
    for key, value in d.items():
        print("  " * indent + f"- {key}: {type(value).__name__}")
        if isinstance(value, dict):
            print_dict_structure(value, indent + 1)

## Evaluation
def find_best_threshold(data, step=0.001,range_=(0, 1.01)):
    best_threshold = 0.0
    best_accuracy = 0.0
    thresholds = np.arange(range_[0], range_[1], step)

    for threshold in thresholds:
        correct = 0
        total = 0

        for item in data:
            true_label = item['novelty'].lower()
            score = item['novelty_score']

            predicted_label = 'y' if score >= threshold else 'n'

            if predicted_label == true_label:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy

def evaluate_threshold(data, threshold):

    y_true = [x['novelty'].lower() for x in data]
    y_pred = ['y' if x['novelty_score']>= threshold else 'n' for x in data]
    print_metrics(y_true,y_pred)

## Shibuyama 2021: average vector distance of citation titles
def preprocess_data_for_shibuyama_0519(test_data_, data,target_key='hypothesis_4omini'):
    ref = data['ref']
    seed = data['seeds']
    df_negs = pd.concat([
        data['seed_negative_paraphrase'],
        data['seed_negative_subset'],
        data['seed_negative_combination_expanded']
    ])
    processed = {}
    for s in test_data_.keys():
        processed[s] = []
        print(f"[{s}] processing {len(test_data_[s])} samples")
        for item in tqdm(test_data_[s]):
            if item['novelty'] == 'Y':
                corpus = ref[ref['cited_by'] == item['id']][target_key].tolist()
                keywords = seed[seed['paperId'] == item['id']]['keywords_4omini'].unique().tolist()
                keywords = safe_eval_list(keywords[0]) if keywords else []
            else:
                original = item['original']
                original_id = ref[ref['hypothesis_4omini'].isin(original)]['paperId'].unique().tolist()
                if not original_id:
                    continue
                corpus = ref[ref['cited_by'].isin(original_id)][target_key].tolist()
                keywords = df_negs[df_negs['sentence'] == item['given_idea']]['idea_keywords_llama'].unique().tolist()
                keywords = safe_eval_list(keywords[0]) if keywords else []

            if not keywords or not corpus:
                continue
            item['keywords'] = keywords
            item['corpus'] = corpus
            processed[s].append(item)
        print(f"[{s}] kept {len(processed[s])} items after filtering")
    return processed

def compute_novelty_shibuyama2021(reference_titles, q_percentile=100):
    """
    Compute the novelty score for a document based on its cited reference titles.

    Parameters:
    - reference_titles (List[str]): A list of titles of cited documents.
    - q_percentile (int): The percentile to use for novelty (e.g., 100 means max distance).

    Returns:
    - novelty_score (float): The novelty score based on q-th percentile of pairwise cosine distances.

    Parameters:
    - reference_titles (List[str]): A list of titles of cited documents.
    - q_percentile (int): The percentile to use for novelty (e.g., 100 means max distance).

    Returns:
    - novelty_score (float): The novelty score based on q-th percentile of pairwise cosine distances.
    """
    # Load scispaCy model (you can install it via: pip install https://s3.amazonaws.com/allenai-scispacy/releases/en_core_sci_lg-0.4.0.tar.gz)
    nlp = spacy.load("en_core_web_lg")  # 200-dim biomedical word embeddings
    # Step 1: Compute average vector for each reference
    vectors = []
    for title in reference_titles:
        doc = nlp(title)
        word_vectors = [token.vector for token in doc if token.has_vector and not token.is_stop]
        if word_vectors:
            vectors.append(np.mean(word_vectors, axis=0))
    
    if len(vectors) < 2:
        return 0.0  # Not enough references to compute pairwise distance
    
    # Step 2: Compute cosine distances between all reference pairs
    distances = []
    for vec1, vec2 in combinations(vectors, 2):
        dist = cosine(vec1, vec2)
        distances.append(dist)
    
    # Step 3: Compute novelty score as q-th percentile
    novelty_score = np.percentile(distances, q_percentile)
    return novelty_score

def run_shibuyama2021(data,raw_data):
    processed = preprocess_data_for_shibuyama_0519(raw_data, data)
    results_shibuyama = {}
    for s in processed.keys():
        results_shibuyama[s] = []
        for i, item in enumerate(tqdm(processed[s],desc=f'Processing {s} set')):

            # print(item['corpus'])
            # print('-'*100)
            corpus = item['corpus']
            # if not corpus or (item['novelty'] == 'N' and gold == item['gold']):
            #     continue
            # else:
            gold = item['novelty']
            corpus = [title.replace('"', '') for title in corpus]
            novelty_score = compute_novelty_shibuyama2021(corpus,q_percentile=90)
            item['novelty_score'] = novelty_score
            # print(f'gt novelty: {item['novelty']}')
            # print(f'len(corpus): {len(corpus)}')
            # print(f'')

            # print(f'Novelty score: {novelty_score}')
            # print('-'*100)
            results_shibuyama[s].append(item)

    if best_threshold is None:
        print(f"Start Evaluation | Finding Best Threshold | ")
        best_threshold, best_accuracy = find_best_threshold(results_shibuyama['train'])
    else:
        print(f"Start Evaluation | Using Predefined Best Threshold: {best_threshold:.5f}")
        best_threshold = 0.2
    print(f"Start Evaluation | Best threshold: {best_threshold:.5f}, Best accuracy: {best_accuracy:.5f}")
    evaluate_threshold(results_shibuyama['test'], best_threshold)
        
## Uzzi 2013 Z-score Atypicality
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations
import random
from tqdm import tqdm

def uzzi_build_keyword_graph(keyword_lists):
    G = nx.Graph()
    for kw_list in keyword_lists:
        for kw1, kw2 in combinations(set(kw_list), 2):
            if G.has_edge(kw1, kw2):
                G[kw1][kw2]['weight'] += 1
            else:
                G.add_edge(kw1, kw2, weight=1)
    return G

def uzzi_generate_randomized_network(G):
    G_rand = G.copy()
    edges = list(G_rand.edges())
    tries = 0
    max_tries = len(edges) * 10

    while tries < max_tries:
        tries += 1
        (u1, v1), (u2, v2) = random.sample(edges, 2)
        if len({u1, v1, u2, v2}) < 4:
            continue
        if random.random() < 0.5:
            a, b = u1, v2
            c, d = u2, v1
        else:
            a, b = u1, u2
            c, d = v1, v2

        if not G_rand.has_edge(a, b) and not G_rand.has_edge(c, d):
            G_rand.remove_edge(u1, v1)
            G_rand.remove_edge(u2, v2)
            G_rand.add_edge(a, b)
            G_rand.add_edge(c, d)
            edges = list(G_rand.edges())

    return G_rand

def uzzi_score_ideas(idea_keywords, z_score_dict):
    scores = []
    for kws in idea_keywords:
        z_list = []
        for kw1, kw2 in combinations(set(kws), 2):
            pair = tuple(sorted([kw1, kw2]))
            if pair in z_score_dict:
                z_list.append(z_score_dict[pair])
        if z_list:
            score = -np.percentile(z_list, 10)
        else:
            score = 0
        scores.append(score)
    return scores

def zscore_atypicality_data_preprocessing_uzzi(data,raw_data):
    df_all = data['ref']
    df_all.drop_duplicates(subset=['keywords_4omini'],inplace=True)
    df_all['keywords_4omini'] = df_all['keywords_4omini'].apply(safe_eval_list)

    all_keywords = []
    for i,row in df_all.iterrows():
        row_keywords = [x.strip() for x in row['keywords_4omini']]
        all_keywords.append(row_keywords)

    idea_keywords = [x['idea_keywords_llama'] for x in raw_data['train']]
    print(f'len(all_keywords): {len(all_keywords)},len(idea_keywords): {len(idea_keywords)}')
    return all_keywords, idea_keywords

# Fast approximation version of build_zscore_table using configuration model expectation
# No rewiring simulation required — uses degree product estimation
def build_zscore_table_fast_uzzi(reference_keywords):
    from collections import defaultdict
    import numpy as np
    from itertools import combinations

    # Build co-occurrence count and node strengths
    pair_counts = defaultdict(int)
    node_strength = defaultdict(int)
    total_weight = 0

    for kw_list in reference_keywords:
        kws = set(kw_list)
        for i, j in combinations(kws, 2):
            pair = tuple(sorted((i, j)))
            pair_counts[pair] += 1
            node_strength[i] += 1
            node_strength[j] += 1
            total_weight += 1

    z_score_dict = {}
    for (i, j), observed in pair_counts.items():
        si = node_strength[i]
        sj = node_strength[j]
        expected = (si * sj) / total_weight if total_weight > 0 else 0
        std = np.sqrt(expected * (1 - expected / total_weight)) if total_weight > 0 else 1
        z = (observed - expected) / std if std > 0 else 0
        z_score_dict[(i, j)] = z

    return z_score_dict

def run_uzzi_zscore_atypicality(data,raw_data,best_threshold=None):
    all_keywords, idea_keywords = zscore_atypicality_data_preprocessing_uzzi(data,raw_data)

    z_score_dict = build_zscore_table_fast_uzzi(list(all_keywords))
    uzzi_results = {}

    for s in raw_data.keys():
        uzzi_results[s] = []
        idea_keywords= [x['idea_keywords_llama'] for x in raw_data[s]]
        scores = uzzi_score_ideas(idea_keywords, z_score_dict)
        for (item,score) in zip(raw_data[s],scores):
            item.pop('corpus',None)
            item.pop('idea_keywords_llama',None)

            item['novelty_score'] = score
            uzzi_results[s].append(item)


    if best_threshold is None:
        print(f"Start Evaluation | Finding Best Threshold | ")
        best_threshold, best_accuracy = find_best_threshold(uzzi_results['train'])
        print(f"Best threshold: {best_threshold:.5f}, Best accuracy: {best_accuracy:.5f}\n\n")

    else:
        print(f"Start Evaluation | Using Predefined Best Threshold: {best_threshold:.5f}")
        # best_threshold = 0.2
    evaluate_threshold(uzzi_results['test'], best_threshold)

    # print(f"Start Evaluation | Best threshold: {best_threshold:.5f}, Best accuracy: {best_accuracy:.5f}")


## Liu 2022 novel pairs
def run_liu_2022_novel_pairs(data, raw_data):
    prepared_data = {}
    df_all = data['ref']


    for s in raw_data.keys():
        target_df=pd.DataFrame.from_dict(raw_data[s])
        # target_df.rename(columns={'idea_keywords_llama':'entities'},inplace=True)
        # 合并文本后统一提取关键词
        all_texts = list(df_all['hypothesis_4omini']) + list(target_df['given_idea'])
        all_entities = liu_2022_extract_entities_tfidf(all_texts)

        # 回填到各自的数据集中
        df_all['entities_new'] = all_entities[:len(df_all)]
        target_df['entities_new'] = all_entities[len(df_all):]

        corpus_distances_new = liu_2022_compute_global_entity_distances(df_all['entities_new'])
        novel_pairs_new = liu_2022_get_novel_pairs(corpus_distances_new, percentile=90)
        
        target_df = liu_2022_compute_novelty_scores(target_df, novel_pairs_new)
        prepared_data[s] = target_df.to_dict(orient='records')

    return prepared_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from itertools import combinations
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ----------------------
# Step 1: Extract entities using TF-IDF
# ----------------------

def liu_2022_extract_entities_tfidf(texts, top_k=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    entities_list = []
    for row in X:
        sorted_indices = np.argsort(row.toarray()[0])[::-1]
        top_entities = feature_names[sorted_indices[:top_k]]
        entities_list.append(list(top_entities))
    return entities_list


# ----------------------
# Step 2: Compute pairwise cosine distance using SBERT
# ----------------------

def liu_2022_compute_global_entity_distances(corpus_entities):
    all_entity_pairs = set()
    entity_embedding_cache = {}

    # Build all unique pairs
    for entities in corpus_entities:
        for e1, e2 in combinations(set(entities), 2):
            pair = tuple(sorted((e1, e2)))
            all_entity_pairs.add(pair)

    unique_entities = list(set(itertools.chain(*all_entity_pairs)))
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(unique_entities, convert_to_numpy=True)
    entity_embedding_cache = dict(zip(unique_entities, embeddings))

    # Compute distances
    distances = []
    for e1, e2 in tqdm(all_entity_pairs, desc="Calculating distances"):
        v1 = entity_embedding_cache[e1].reshape(1, -1)
        v2 = entity_embedding_cache[e2].reshape(1, -1)
        dist = cosine_distances(v1, v2)[0][0]
        distances.append(((e1, e2), dist))

    return distances


def liu_2022_get_novel_pairs(distances, percentile=90):
    dist_values = np.array([d for (_, d) in distances])
    threshold = np.percentile(dist_values, percentile)
    novel_set = {pair for pair, d in distances if d >= threshold}
    return novel_set


# ----------------------
# Step 3: Score target_df
# ----------------------

def liu_2022_compute_novelty_scores(df, novel_pairs):
    scores = []
    for entities in df['entities_new']:
        entity_pairs = list(combinations(set(entities), 2))
        if not entity_pairs:
            scores.append(0.0)
            continue
        match_count = sum(1 for pair in entity_pairs if tuple(sorted(pair)) in novel_pairs)
        score = match_count / len(entity_pairs)
        scores.append(score)
    df['novelty_score'] = scores
    return df


