import os
import sys
import json
import argparse
import pandas as pd
import torch
import warnings
import gc
from torch.utils.data import DataLoader
from IPython.display import clear_output

# === Suppress Warnings ===
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# === Environment Configuration ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(f'CUDA device count: {torch.cuda.device_count()}')

# === Add Project Path ===
sys.path.append('/home/duy/ly/novelty_checker')

# === Project-Specific Imports ===
import utils.dataset as dataset
import utils.embeddings as embeddings


# === Argument Parsing ===
def parse_args():
    parser = argparse.ArgumentParser(description="Run embedding retrieval experiment")
    parser.add_argument('--data_to_load', type=str, default='marketing', choices=['marketing', 'acl', 'emnlp'])
    parser.add_argument('--embedding_model_name', type=str, default='nli')
    parser.add_argument('--evaluation_mode', type=str, default='retrieval', choices=['retrieval', 'nc'])
    parser.add_argument('--data_types', nargs='+', default=['fine_tuned'])
    parser.add_argument('--neg_types', nargs='+', default=['all', 'paraphrase', 'subset', 'combination'])
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_k_list', nargs='+', type=int, default=[1, 3, 5, 10, 20, 50])
    parser.add_argument('--suffix', type=str, default='0519')
    parser.add_argument('--date', type=str, default='0519')
    parser.add_argument('--version', type=str, default='20250512')
    return parser.parse_args()


# === Root Info Mapping ===
ROOT_INFO = {
    'marketing': {'server': 'duy', 'domain': 'marketing', 'version': '20250512', 'venue': None},
    'acl': {'server': 'duy', 'domain': 'nlp_papers', 'version': '20250512', 'venue': 'acl'},
    'emnlp': {'server': 'duy', 'domain': 'nlp_papers', 'version': '20250218', 'venue': 'emnlp'}
}

# === Model Mapping ===
MODEL_NAME_DICT = {
    "nli": "microsoft/deberta-xlarge-mnli",
    "BGE": "BAAI/bge-base-en-v1.5",
    # Add other models if needed
}


def prepare_test_data(data_to_load, version):
    config = ROOT_INFO[data_to_load]
    data = dataset.load_raw_data(
        server=config['server'],
        domain=config['domain'],
        version=config['version'],
        venue=config['venue'],
        split_ratio='613'
    )
    preprocessed_data_dir = f'/data/duy/ly/{config["domain"]}/versions/{version}'
    data['nc'] = dataset.load_preprocseed_data(
        path_to_data_root=preprocessed_data_dir,
        ref=data['ref'],
        if_add_corpus=False
    )['nc']
    
    processed = dataset.create_alignment_samples_gpt(data, splits=('train', 'val', 'test'), comb_mode=2)
    nc_data_all = processed['test'] + dataset.build_nc_positive_test_samples(
        seed_test=None, df_not_used=data['splits']['not_used'], ref=data['ref']
    )
    to_process = {'test': nc_data_all}
    nc_data_all = dataset.add_corpus(to_process, data['ref'], splits=('test',))['test']
    return nc_data_all


def get_model_dir(embedding_model_key, domain, version):
    return f"./constrastive_model/{MODEL_NAME_DICT[embedding_model_key]}/final_model_{domain}_using_ref/{version}"


def get_save_dir(evaluation_mode, date, data_to_load):
    base = f"/home/duy/ly/novelty_checker/notebooks/clean_version/constrastive_model/{date}/{data_to_load}"
    if evaluation_mode == 'retrieval':
        return f"{base}/retrieved_results/rerun"
    elif evaluation_mode == 'nc':
        return f"{base}/prepared_for_nc"
    else:
        raise ValueError("Invalid evaluation_mode")


def run_retrieval_experiment(test_data, calculated_embeddings, neg_types, evaluation_mode,
                             save_dir, top_k_list, top_k, data_type, data_to_load,
                             embedding_model_name, suffix, verbose=False):

    all_results = pd.DataFrame()

    for neg_type in neg_types:
        if evaluation_mode == 'retrieval':
            test_data_ = [x for x in test_data if (x['novelty'] == 'N')]
        if neg_type == 'all':
            test_data_ = [x for x in test_data if (x['label'] == 1.0)]
        else:
            test_data_ = [x for x in test_data if (x['neg_type'] == neg_type and x['label'] == 1.0)]

        print(f"Evaluating: neg_type={neg_type}, data_type={data_type}, test_size={len(test_data_)}")

        retrieval_log, results = embeddings.prepare_retrieval_results_for_nc_new(
            test_data=test_data_,
            query_emb_dict=calculated_embeddings[data_type]['query_emb_dict'],
            candidate_emb_dict=calculated_embeddings[data_type]['candidate_emb_dict'],
            model_name=data_type,
            evaluation_type='default',
            top_k=top_k,
            save_path=f"{save_dir}/nc_results_{data_to_load}_{embedding_model_name}_{data_type}_{neg_type}_{suffix}.json",
            verbose=verbose,
            top_k_list=top_k_list
        )

        results['embedding_model'] = embedding_model_name
        results['data_type'] = data_type

        with open(f"{save_dir}/evaluation_results_{embedding_model_name}_{data_type}_{neg_type}_{suffix}.json", 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        result_df = pd.DataFrame.from_dict(results, orient='index', columns=[f'{data_to_load}_{data_type}_{neg_type}'])
        all_results = pd.concat([all_results, result_df], axis=1)

    return all_results


def main():
    args = parse_args()

    os.makedirs(args.suffix, exist_ok=True)
    test_data = prepare_test_data(args.data_to_load, args.version)
    print(f"Loaded {len(test_data)} test samples from {args.data_to_load}")

    save_dir = get_save_dir(args.evaluation_mode, args.date, args.data_to_load)
    os.makedirs(save_dir, exist_ok=True)

    model_dir = get_model_dir(args.embedding_model_name, args.data_to_load, args.version)
    calculated_embeddings = embeddings.create_sentence_transformer_embeddings(
        test_data=test_data,
        embedding_model_name=args.embedding_model_name,
        domain=args.data_to_load,
        model_name=None,
        data_types=['fine_tuned', 'vanilla'],
        model_path=model_dir
    )

    full_results = pd.DataFrame()
    for data_type in args.data_types:
        results = run_retrieval_experiment(
            test_data=test_data,
            calculated_embeddings=calculated_embeddings,
            neg_types=args.neg_types,
            evaluation_mode=args.evaluation_mode,
            save_dir=save_dir,
            top_k_list=args.top_k_list,
            top_k=args.top_k,
            data_type=data_type,
            data_to_load=args.data_to_load,
            embedding_model_name=args.embedding_model_name,
            suffix=args.suffix
        )
        full_results = pd.concat([full_results, results], axis=1)

    full_results.to_csv(f"{save_dir}/retrieval_results_{args.data_to_load}_{args.embedding_model_name}_using_ref.csv", index=False)
    print(f"Results saved to {save_dir}/retrieval_results_{args.data_to_load}_{args.embedding_model_name}_using_ref.csv")

    clear_output(wait=True)
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
