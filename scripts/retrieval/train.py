# === Imports ===
import os
import sys
import json
import warnings
import gc
import argparse
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses

from IPython.display import clear_output

# === Configuration ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

print(f"CUDA device count: {torch.cuda.device_count()}")

# === Custom Module Path ===
sys.path.append('/home/duy/ly/novelty_checker')

# === Project-Specific Imports ===
import utils.dataset as dataset
import utils.embeddings as embeddings


# === Constants ===
MODEL_NAME_DICT = {
    "stsb": "stsb-roberta-base",
    "GTE": "thenlper/gte-base",
    "E5": "intfloat/e5-base-v2",
    "BGE": "BAAI/bge-base-en-v1.5",
    "SimCSE": "princeton-nlp/sup-simcse-roberta-base",
    "nli": "microsoft/deberta-xlarge-mnli",
    "sbert": "paraphrase-MiniLM-L6-v2",
    "sbert_all": "all-MiniLM-L6-v2"
}

ROOT_INFO = {
    'acl': {
        'server': 'duy',
        'domain': 'nlp_papers',
        'version': '20250512',
        'venue': 'acl'
    }
}

SUFFIX = 'using_ref'
BATCH_SIZE = 8
EPOCHS = 3
VERSION = '20250512'


# === Functions ===

def prepare_data(data_to_load):
    config = ROOT_INFO[data_to_load]
    data = dataset.load_raw_data(
        server=config['server'],
        domain=config['domain'],
        version=config['version'],
        venue=config['venue'],
        split_ratio='613'
    )
    dataset.check_overlap(data['splits']['train'], data['splits']['val'],
                          data['splits']['test'], data['splits']['not_used'])

    processed = dataset.create_alignmnet_samples_using_ref(data, splits=('train', 'val', 'test'))
    return processed


def train_and_evaluate(model_name, data_to_load, processed_data):
    embedding_model_name = MODEL_NAME_DICT[model_name]

    output_dir = f"./constrastive_model/{embedding_model_name}/final_model_{data_to_load}_{SUFFIX}/{VERSION}"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.jsonl")

    print(f"\n=== Training Model: {embedding_model_name} on {data_to_load} ===")
    print(f"Output Dir: {output_dir}")

    train_data = processed_data['train']
    val_data = processed_data['val']
    print(f"Train Samples: {len(train_data)} | Val Samples: {len(val_data)}")

    train_examples = embeddings.build_input_examples(train_data)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    evaluator = embeddings.build_val_evaluator(val_data)

    model = SentenceTransformer(embedding_model_name)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    train_fixed_epochs_with_test_eval(
        model=model,
        train_dataloader=train_dataloader,
        loss=loss_fn,
        evaluator=evaluator,
        epochs=EPOCHS,
        output_path=output_dir,
        log_path=log_path,
        target_key='binary_cosine_f1'
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()
    clear_output(wait=True)


def train_fixed_epochs_with_test_eval(model, train_dataloader, loss, evaluator,
                                      epochs=10, output_path="./model_output",
                                      log_path="./training_log.jsonl",
                                      target_key='binary_cosine_f1'):
    best_val_score = -1.0
    os.makedirs(output_path, exist_ok=True)

    with open(log_path, "w") as log_file:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            model.fit(
                train_objectives=[(train_dataloader, loss)],
                epochs=1,
                warmup_steps=10,
                show_progress_bar=True,
                output_path=None
            )

            print("Evaluating on validation set...")
            val_result = evaluator(model)
            print("Validation result:", val_result)

            matched_key = [k for k in val_result if target_key in k][0]
            val_score = val_result[matched_key]
            print(f"Validation score ({matched_key}): {val_score:.4f}")

            if val_score > best_val_score:
                print("New best score. Saving model.")
                best_val_score = val_score
                model.save(output_path)
            else:
                print(f"No improvement. Best score so far: {best_val_score:.4f}")

            log_file.write(json.dumps({"epoch": epoch + 1, "val_score": val_score}) + "\n")
            log_file.flush()

    print(f"\nTraining complete. Best model saved to: {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train contrastive sentence embedding model")
    parser.add_argument('--model', type=str, default='nli', help='Model key from MODEL_NAME_DICT')
    parser.add_argument('--dataset', type=str, default='acl', help='Dataset key from ROOT_INFO')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='./constrastive_model/', help='Output directory root')
    parser.add_argument('--suffix', type=str, default='using_ref', help='Suffix for model subfolder')
    parser.add_argument('--version', type=str, default='20250512', help='Version name for dataset/model folder')
    parser.add_argument('--target_key', type=str, default='binary_cosine_f1', help='Target metric for model selection')
    return parser.parse_args()

# === Main Entry ===
def main():
    args = parse_arguments()

    processed_data = prepare_data(args.dataset)

    embedding_model_name = MODEL_NAME_DICT[args.model]
    output_dir = os.path.join(args.output_dir, embedding_model_name, f"final_model_{args.dataset}_{args.suffix}", args.version)
    log_path = os.path.join(output_dir, "training_log.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    train_data = processed_data['train']
    val_data = processed_data['val']

    train_examples = embeddings.build_input_examples(train_data)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    evaluator = embeddings.build_val_evaluator(val_data)

    model = SentenceTransformer(embedding_model_name)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    train_fixed_epochs_with_test_eval(
        model=model,
        train_dataloader=train_dataloader,
        loss=loss_fn,
        evaluator=evaluator,
        epochs=args.epochs,
        output_path=output_dir,
        log_path=log_path,
        target_key=args.target_key
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()
    clear_output(wait=True)

if __name__ == "__main__":
    main()

# python train_model.py --model nli --dataset acl --epochs 5 --batch_size 16 --suffix test_run --version 20250523
