
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
import numpy as np
from sentence_transformers.util import cos_sim
import torch

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def build_input_examples(data):
    return [
        InputExample(texts=[item["given_idea"], item["original"]])
        for item in data if float(item["label"]) == 1.0
    ]

def build_val_evaluator(data, name="val"):
    sentences1 = [item["given_idea"] for item in data]
    sentences2 = [item["original"] for item in data]
    labels = [float(item["label"]) for item in data]
    return BinaryClassificationEvaluator(sentences1, sentences2, labels, name='val-binary')


def train_fixed_epochs_with_test_eval(model, train_dataloader, loss, evaluator,
                                      epochs=10, output_path="./model_output", log_path="./training_log_25032025.jsonl",target_key='binary_cosine_f1'):

    best_val_score = -1.0
    min_loss = float('inf')
    os.makedirs(output_path, exist_ok=True)
    log_file = open(log_path, "w")

    for epoch in range(epochs):
        print(f"\n🔁 Epoch {epoch + 1}/{epochs}")

        # === Training one epoch ===
        model.fit(
            train_objectives=[(train_dataloader, loss)],
            epochs=1,
            warmup_steps=10,
            show_progress_bar=True,
            output_path=None  # don't save intermediate
        )

        # === Evaluate on test set ===
        print("🧪 Evaluating on test set...")
        val_result = evaluator(model)
        print("💡 Val result:")
        print(val_result)
        target_key_  = [x for x in val_result if target_key in x][0]
        val_score = val_result[target_key_]
        print(f'val_score: {val_score}')
        # === Save best model ===
        if val_score > best_val_score:
            print(f"✅ New best val score: {val_score} (prev: {best_val_score}) → Saving model.")
            best_val_score = val_score
            model.save(output_path)
        else:
            print(f"📉 Val score: {val_score} did not improve (best: {best_val_score})")

        # === Log this epoch ===
        log_entry = {
            "epoch": epoch + 1,
            "val_score": val_score
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()
        print('-'*100)
        print()

    log_file.close()
    print("\n🏁 Training complete. Best model saved to:", output_path)


def evaluate_retrieval_cached(model, test_data, top_k=10):
    print("🧠 Step 1: Collecting all unique sentences for encoding...")

    all_queries = set()
    all_candidates = set()

    for item in test_data:
        all_queries.add(item["given_idea"])
        all_candidates.update(item["corpus"])

    print(f"→ {len(all_queries)} unique queries")
    print(f"→ {len(all_candidates)} unique candidates")

    print("⚙️ Step 2: Encoding queries & candidates...")
    all_queries = list(all_queries)
    all_candidates = list(all_candidates)

    query_embeddings = model.encode(all_queries, convert_to_tensor=True, show_progress_bar=True)
    candidate_embeddings = model.encode(all_candidates, convert_to_tensor=True, show_progress_bar=True)

    # Build lookup dicts
    query_emb_dict = dict(zip(all_queries, query_embeddings))
    candidate_emb_dict = dict(zip(all_candidates, candidate_embeddings))

    print("🚀 Step 3: Running retrieval per query...")

    hits_at_k = [0] * top_k
    reciprocal_ranks = []
    average_precisions = []

    for item in tqdm(test_data, desc="Evaluating"):
        query = item["given_idea"]
        gold = item["original"]
        candidates = item["corpus"]

        if gold not in candidates:
            print(f"[WARN] Gold not in candidates: {gold[:50]}...")
            continue

        # get embeddings
        query_emb = query_emb_dict[query]
        candidate_embs = torch.stack([candidate_emb_dict[c] for c in candidates])

        # compute scores
        scores = cos_sim(query_emb, candidate_embs)[0]
        top_results = torch.topk(scores, k=min(top_k, len(candidates)))

        found = False
        precisions = []

        for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices), start=1):
            retrieved = candidates[idx]
            is_correct = int(retrieved == gold)

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

    cumulative_hits = np.cumsum(hits_at_k)
    num_queries = len(test_data)

    for k in [1, 3, 5, 10]:
        if k <= top_k:
            acc_k = cumulative_hits[k - 1] / num_queries
            print(f"✅ Accuracy@{k}: {acc_k:.4f}")

    print(f"📈 MRR@{top_k}: {np.mean(reciprocal_ranks):.4f}")
    print(f"📊 MAP@{top_k}: {np.mean(average_precisions):.4f}")


def build_sentence_transformer_embeddings(all_queries, all_candidates, model_name=None,embedding_model_name=None,model_type='vanilla',model_dir=None):
    def build_sbert_embeddings(corpus, model=None):
        if model is None:
            model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        embeddings = model.encode(corpus, convert_to_tensor=True,show_progress_bar=True)
        return embeddings
    
    if embedding_model_name is None:
        embedding_model_name = 'sbert'

    if model_name is None:
        model_name = {
        "stsb": "stsb-roberta-base",
        "GTE": "thenlper/gte-base",
        "E5": "intfloat/e5-base-v2",
        "BGE": "BAAI/bge-base-en-v1.5",
        "SimCSE": "princeton-nlp/sup-simcse-roberta-base",
        "nli": "microsoft/deberta-xlarge-mnli",
        "sbert": "paraphrase-MiniLM-L6-v2",
        "sbert_all": "all-MiniLM-L6-v2",
        }

    if model_dir is not None:
        print(f'Loading model from {model_dir}')
        model = SentenceTransformer(model_dir)
    else:
        print(f'Initializing model from {model_name[embedding_model_name]}')
        model = SentenceTransformer(model_name[embedding_model_name])

    query_embeddings = build_sbert_embeddings(all_queries,model)
    candidate_embeddings = build_sbert_embeddings(all_candidates,model)

    # Build lookup dicts
    query_emb_dict = dict(zip(all_queries, query_embeddings))
    candidate_emb_dict = dict(zip(all_candidates, candidate_embeddings))
    return query_emb_dict, candidate_emb_dict

# import torch
# import numpy as np
# import json
# from tqdm import tqdm
# from sentence_transformers.util import cos_sim
from datetime import datetime

def prepare_retrieval_results_for_nc_new(
    test_data,
    query_emb_dict,
    candidate_emb_dict,
    model_name="vanilla",
    evaluation_type="default",  # 'default' or 'two_hits'
    top_k=10,
    top_k_list=[1, 3, 5, 7, 10],
    save_path=None,
    verbose=False
):  
    if verbose:
        print("🚀 Step 3: Running retrieval per query...")

    hits_at_k = {k: 0 for k in top_k_list}
    reciprocal_ranks = []
    average_precisions = []
    retrieval_log = []
    count_two_hits = 0
    valid_queries = 0

    for item in tqdm(test_data, desc=f"Evaluating ({model_name})"):
        query = item["given_idea"]
        gold = item["original"]
        candidates = item["corpus"]
        novelty = item.get("novelty", None)
        neg_type = item.get("neg_type", "NA")
        date = item.get("publicationDate", "NA")

        # Ensure gold is a list
        gold_set = {gold} if isinstance(gold, str) else set(gold)
        if not gold_set.intersection(set(candidates)):
            candidates.append(gold)
            if verbose:
                print(f"[WARN] Skipping: gold answer not in candidates → {gold_set}")
            continue

        try:
            query_emb = query_emb_dict[query]
            candidate_embs = torch.stack([candidate_emb_dict[c] for c in candidates])
        except KeyError as e:
            if verbose:
                print(f"[ERROR] Missing embedding for: {e}")
            continue

        scores = cos_sim(query_emb, candidate_embs)[0]
        top_results = torch.topk(scores, k=min(top_k, len(candidates)))
        retrieved_candidates = [candidates[idx] for idx in top_results.indices]

        # Evaluation record
        result_item = {
            "id": item['id'],
            "given_idea": query,
            "original": list(gold_set),
            "neg_type": neg_type,
            "publicationDate": date.strftime('%Y-%m-%d') if not isinstance(date, str) else date,
            "novelty": novelty,
            "top_k": [],
            "is_correct_retrieval": False
        }

        # Evaluation metrics
        rr = 0.0
        precisions = []
        retrieved_hits = []

        for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices), start=1):
            candidate = candidates[idx]
            is_hit = candidate in gold_set
            result_item["top_k"].append({
                "rank": rank,
                "sentence": candidate,
                "score": float(score),
                "hit": is_hit
            })
            if is_hit:
                if rr == 0.0:
                    rr = 1.0 / rank
                precisions.append(len(precisions) + 1 / rank)
                retrieved_hits.append(candidate)
            else:
                retrieved_hits.append(None)

        reciprocal_ranks.append(rr)
        average_precisions.append(np.mean(precisions) if precisions else 0.0)
        valid_queries += 1

        # Accuracy@k
        for k in top_k_list:
            retrieved_top_k = retrieved_candidates[:k]
            if evaluation_type == 'default':
                if any(r in gold_set for r in retrieved_top_k):
                    hits_at_k[k] += 1
            elif evaluation_type == 'two_hits':
                hits = set(r for r in retrieved_top_k if r in gold_set)
                if len(hits) == len(gold_set):
                    hits_at_k[k] += 1
                    if k == max(top_k_list):
                        count_two_hits += 1

        # Save hit status
        result_item["is_correct_retrieval"] = (
            any(r in gold_set for r in retrieved_candidates)
            if evaluation_type == 'default'
            else len([r for r in retrieved_hits if r is not None]) == len(gold_set)
        )
        retrieval_log.append(result_item)

    # Save output JSON
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./retrieval_results_{model_name}_{evaluation_type}_{timestamp}.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_log, f, indent=2, ensure_ascii=False)


    print(f"\n✅ Saved detailed retrieval results to: {save_path}")
    print(f"📄 Evaluated on {valid_queries} valid queries")

    # Report metrics
    results = {}
    for k in top_k_list:
        acc_k = hits_at_k[k] / valid_queries if valid_queries > 0 else 0.0
        results[f"Accuracy@{k}"] = acc_k
        if verbose:
            print(f"✅ Accuracy@{k}: {acc_k}")

    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    map_score = np.mean(average_precisions) if average_precisions else 0.0
    if verbose:
        print(f"📈 MRR: {mrr}")
        print(f"📊 MAP: {map_score}")
        print(f"🎯 Full hits@{max(top_k_list)} (two_hits only): {count_two_hits}")

    results.update({
        "MRR": mrr,
        "MAP": map_score,
        "count_two_hits": count_two_hits
    })

    return retrieval_log, results


def prepare_retrieval_results_for_nc(
    test_data,
    query_emb_dict,
    candidate_emb_dict,
    model_name="vanilla",
    evaluation_type="default",
    top_k=10,
    save_path=None,
    verbose=False,
    top_k_list=[1, 3, 5, 7, 10],
    type_evaluate='two_hits'
):
    print("🚀 Step 3: Running retrieval per query...")

    hits_at_k = {k: 0 for k in top_k_list}
    reciprocal_ranks = []
    average_precisions = []
    retrieval_log = []
    count_two_hits = 0

    for item in tqdm(test_data, desc=f"Evaluating ({model_name})"):
        query = item["given_idea"]
        gold = item["original"]
        if isinstance(gold, str):
            gold = [gold]  # ensure it's a list

        candidates = item["corpus"]
        novelty = item.get("novelty", None)

        try:
            query_embedding = query_emb_dict[query]
            candidate_embeddings = torch.stack([candidate_emb_dict[c] for c in candidates])
        except KeyError as e:
            if verbose:
                print(f"[ERROR] Missing embedding for: {e}")
            continue

        scores = cos_sim(query_embedding, candidate_embeddings)[0]
        top_results = torch.topk(scores, k=min(top_k, len(candidates)))
        date = pd.Timestamp(item['publicationDate']).strftime('%Y-%m-%d') if 'publicationDate' in item else "NA"

        result_item = {
            "given_idea": query,
            "original": gold,
            "neg_type": item.get("neg_type", "NA"),
            "publicationDate": date,
            "novelty": novelty,
            "is_correct_retrieval": False,
            "top_k": []
        }

        reciprocal_rank = 0.0
        precisions = []
        retrieved_hits = []

        for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices), start=1):
            retrieved = candidates[idx]
            is_correct = int(retrieved in gold)
            retrieved_hits.append(retrieved if is_correct else None)

            result_item["top_k"].append({
                "rank": rank,
                "sentence": retrieved,
                "score": float(score),
                "hit": is_correct
            })

            if is_correct:
                if reciprocal_rank == 0.0:
                    reciprocal_rank = 1.0 / rank
                precisions.append(1.0 / rank)

        reciprocal_ranks.append(reciprocal_rank)
        average_precisions.append(np.mean(precisions) if precisions else 0.0)
        # count_two_hits = 0
        for k in top_k_list:
            retrieved_top_k = [candidates[top_results.indices[i]] for i in range(min(k, len(candidates)))][:k]

            if type_evaluate == 'default':
                if any(r in gold for r in retrieved_top_k):
                    hits_at_k[k] += 1
            elif type_evaluate == 'two_hits':
                hits = list(set([r for r in retrieved_top_k if r in gold]))
                
                if len(hits) == len(gold):  # all gold retrieved
                    hits_at_k[k] += 1
                    if k == 50:
                        count_two_hits += 1
                    # if k == top_k:
                    # count_two_hits += 1
                    # print(f'hits: {len(hits)}, len(gold): {len(gold)}')

        result_item['is_correct_retrieval'] = (
            any(r in gold for r in retrieved_hits)
            if type_evaluate == 'default'
            else len([r for r in retrieved_hits if r is not None]) == len(gold)
        )

        retrieval_log.append(result_item)

    if save_path is None:
        save_path = f"./retrieval_results_{model_name}_{evaluation_type}.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_log, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved detailed retrieval results to {save_path}")

    num_queries = len(test_data)
    results = {}

    for k in top_k_list:
        acc_k = hits_at_k[k] / num_queries
        print(f"✅ Accuracy@{k}  {acc_k}")
        results[f"Accuracy@{k}"] = acc_k

    mrr = np.mean(reciprocal_ranks)
    map_score = np.mean(average_precisions)
    print(f"📈 MRR  {mrr}")
    print(f"📊 MAP  {map_score}")
    print(f"🎯 count_two_hits: {count_two_hits}")

    results.update({
        "MRR": mrr,
        "MAP": map_score,
        "count_two_hits": count_two_hits
    })

    return retrieval_log, results


def create_sentence_transformer_embeddings(test_data,embedding_model_name,domain,model_name=None,data_types=['vanilla','fine_tuned'],model_path=None):
    if not model_name:
        model_name = {
                    "stsb": "stsb-roberta-base",
                    "GTE": "thenlper/gte-base",
                    "E5": "intfloat/e5-base-v2",
                    "BGE": "BAAI/bge-base-en-v1.5",
                    "SimCSE": "princeton-nlp/sup-simcse-roberta-base",
                    "nli": "microsoft/deberta-xlarge-mnli",
                    "sbert": "paraphrase-MiniLM-L6-v2",
                    "sbert_all": "all-MiniLM-L6-v2",
                    }
    calculated_embeddings={}
    
    all_queries = set()
    all_candidates = set()
    # test_data = test_data_with_corpus['test']

    for item in test_data:
        all_queries.add(item["given_idea"])
        all_candidates.update(item["corpus"]+['NA']+item['original'])

    print(f"→ {len(all_queries)} unique queries")
    print(f"→ {len(all_candidates)} unique candidates")

    # print("⚙️ Step 2: Encoding queries & candidates...")
    all_queries = list(all_queries)
    all_candidates = list(all_candidates)

    for data_type in data_types:
        if data_type == 'vanilla':

            print(f'Initialzing sentence transformer model...')
            model = SentenceTransformer(model_name[embedding_model_name])          


        elif data_type == 'fine_tuned':
            print(f'Loading model from: {model_path}')
            model= SentenceTransformer(model_path)
            
        print(f'{data_type} model loaded!\n')
        print(f'Creating embeddings | embdd model: {embedding_model_name} | domain: {domain} | data_type: {data_type}\n\n')
        query_embeddings = model.encode(all_queries, convert_to_tensor=True, show_progress_bar=True)
        candidate_embeddings = model.encode(all_candidates, convert_to_tensor=True, show_progress_bar=True)
        # Build lookup dicts
        query_emb_dict = dict(zip(all_queries, query_embeddings))
        candidate_emb_dict = dict(zip(all_candidates, candidate_embeddings))

        calculated_embeddings[data_type]={}
        calculated_embeddings[data_type]['query_emb_dict']=query_emb_dict
        calculated_embeddings[data_type]['candidate_emb_dict']=candidate_emb_dict
    return calculated_embeddings

def success_retreival(top_k_retrieval,top_k):
    hits = [x['hit'] for x in top_k_retrieval[:top_k]] 
    if True in hits:
        return True
    else: return False


def select_samples(data_vanilla,data_ft):
    # Check if two are same
    unique_v = set([x['query'] for x in data_vanilla])
    unique_ft = set([x['query'] for x in data_ft])
    difference = unique_v - unique_ft
    if len(difference) == 0:
        print(f'Same set of queries')
    else:
        print(f'Error! Check if the data has been loaded correctly')
        return None
    
    # bad in v
    bad_in_v_n = [x['query'] for x in data_vanilla if x['gold'] == 'n' and x['new_rule']=='y']
    bad_in_v_p = [x['query'] for x in data_vanilla if x['gold'] == 'y' and x['new_rule']=='n']
    print(f'Vanilla | number of failed examples: N->{len(bad_in_v_n)}, Y->{len(bad_in_v_p)}')

    # bad in v
    good_in_ft_n = [x['query'] for x in data_ft if x['query'] in bad_in_v_n and x['new_rule']=='n']
    good_in_ft_p = [x['query'] for x in data_ft if x['query'] in bad_in_v_p and x['new_rule']=='y']
    print(f'FT | number of success examples: N->{len(good_in_ft_n)}, Y->{len(good_in_ft_p)}')

    
    return good_in_ft_n,good_in_ft_p


def run_retrieval_experiment(test_data,calculated_embeddings,neg_types,evaluation_mode,save_dir,top_k_list,sufix=None,verbose=True,data_type='vanilla',data_to_load='acl',embedding_model_name='sbert',top_k=50, suffix=None):

    for neg_type in neg_types:
        if neg_type == 'all':
            test_data_ = [x for x in test_data if (x['label'] == 1.0)]
        else:
            test_data_ = [x for x in test_data if (x['neg_type'] == neg_type and x['label'] == 1.0)]

        if evaluation_mode == 'retrieval':
            test_data_ = [x for x in test_data if (x['novelty'] == 'N')]

        # test_data = [x for x in test_data_nc if x['label'] == 1.0]
        print(f'evaluation_mode: {evaluation_mode} | neg_type: {neg_type} | len(test_data): {len(test_data_)}\n\n')

        if not suffix:
            suffix = str(len(test_data))

        retrieval_log,results = prepare_retrieval_results_for_nc_new(
            test_data=test_data_,
            query_emb_dict=calculated_embeddings[data_type]['query_emb_dict'],
            candidate_emb_dict=calculated_embeddings[data_type]['candidate_emb_dict'],
            model_name=data_type,
            evaluation_type='default',
            top_k=top_k,
            save_path=f'{save_dir}/nc_results_{data_to_load}_{embedding_model_name}_{data_type}_{sufix}.json',
            verbose=verbose,
            top_k_list = top_k_list
        )
        results['embedding_model']=embedding_model_name
        results['data_type']=data_type

        with open(f'{save_dir}/evaluation_results_{embedding_model_name}_{data_type}_{suffix}.json', 'w+') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print('-'*100, '\n\n')
    
    return retrieval_log,results