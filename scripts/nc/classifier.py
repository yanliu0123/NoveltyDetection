# === Standard Library Imports ===
import os
import sys
import re
import time
import json
import random
import datetime
import warnings

# === Third-Party Imports ===
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# === Warning Configuration ===
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# === Environment Configuration ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')

# === Project-Specific Imports ===
sys.path.append('/home/duy/ly/novelty_checker')

import utils.general as general
import utils.csv_processing as csv_processing
import utils.pdf_processing as pdf_processing
import utils.llm as llm
import utils.paper_search as paper_search
import utils.dataset as dataset
import utils.data_preprocessing as data_preprocessing
import utils.embeddings as embeddings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from typing import List, Dict, Tuple

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

import pandas as pd

def count_specific_scores(score_lists, target_scores=['0', '0.3', '0.5', '0.7', '1.0']):
    """
    统计每个子列表中指定分数出现的次数。

    参数：
        score_lists: List[List[float]]
            每个元素是一个分数列表
        target_scores: List[str]
            要统计的分数值（字符串形式，默认统计 '0', '0.3', '0.5', '0.7', '1.0'）

    返回：
        pd.DataFrame
            每行对应一个子列表中各指定分数的出现次数
    """
    count_df = pd.DataFrame(columns=target_scores)

    for score in score_lists:
        new_record = {}
        for s in target_scores:
            new_record[s] = score.count(float(s))
        count_df = pd.concat([count_df, pd.DataFrame([new_record])], ignore_index=True)

    return count_df

data_to_load = 'acl'
llm_name = 'deepseek'
data_suffix = '100_for_test'
top_k =10
embedding_model_name = 'BGE'
mode = 'manual'
decision_rules = ['llm','rules','new_rule']
decision_rules = [decision_rules[2]]


data_types=['fine_tuned','vanilla']
all_results = {}
for data_type in data_types:
    for dr in decision_rules:
        path_to_result = f'/data/duy/ly/{data_to_load}/versions/20250512/nc_results/0519/{llm_name}/nc_test_{data_to_load}_{embedding_model_name}_{data_type}_{llm_name}_top_{top_k}_{mode}_{data_suffix}.json'
        all_results[data_type] = {}
        print(f'data type: {data_to_load} {data_type} {llm_name} | top_k: {top_k} | decision rule: {dr} | ')
        all_results[data_type]= eval_results(path_to_result,mode=dr)
        print('-'*100,'\n')


from xgboost import train

# data_to_load = 'marketing'
# embedding_model_name='BGE'
data_suffix_train = '100_for_train'
k=5
# llm_name='deepseek'
# top_k=10
# mode='manual'
# data_suffix='100_for_test'
def load_train(data_suffix_train,data_types):
    # data_types=['fine_tuned']
    all_results = {}
    for data_type in data_types:
        for dr in decision_rules:
            path_to_result = f'/data/duy/ly/{data_to_load}/versions/20250512/nc_results/0519/{llm_name}/nc_test_{data_to_load}_{embedding_model_name}_{data_type}_{llm_name}_top_{top_k}_{mode}_{data_suffix_train}.json'
            all_results[data_type] = {}
            print(f'data type: {data_to_load} {data_type} {llm_name} | top_k: {top_k} | decision rule: {dr} | ')
            all_results[data_type]= eval_results(path_to_result,mode=dr)
            print('-'*100,'\n')
    return all_results

def success_retrieval(x, k): 
    # hits = [s['hit'] for s in top_k[:k]]
    # hits = [s['hit'] for s in x['top_k'][:k]]
    return True# in hits

def clean_and_pad(data):
    result = []
    for item in data:
        # print(item)
        label, scores, if_hit= item
        # Skip entries where scores is an empty list
        if scores == []:
            continue
        # Pad if not length 10
        # if len(scores) < 10:
        #     scores = scores + [0] * (10 - len(scores))
        result.append([label, scores,if_hit])
    return result
all_results_train = load_train(data_suffix_train,data_types)

if 'fine_tuned' in all_results.keys():
    ft_n_n = [x['scores'] for x in all_results['fine_tuned'] if x['gold'] == 'n' and x['rules_pred'] == 'n']
    ft_n_y = [x['scores'] for x in all_results['fine_tuned'] if x['gold'] == 'n' and x['rules_pred'] == 'y']
    ft_y_n = [x['scores'] for x in all_results['fine_tuned'] if x['gold'] == 'y' and x['rules_pred'] == 'n']
    ft_y_y = [x['scores'] for x in all_results['fine_tuned'] if x['gold'] == 'y' and x['rules_pred'] == 'y']
    
    ft_train = [[x['gold'],x['scores'][:k],success_retrieval(x,k)] for x in all_results_train['fine_tuned']]
    ft = [[x['gold'],x['scores'][:k],success_retrieval(x,k)] for x in all_results['fine_tuned']]
    ft_train = clean_and_pad(ft_train)
    ft = clean_and_pad(ft)


if 'vanilla' in all_results.keys():
    v_n_n = [x['scores'] for x in all_results['vanilla'] if x['gold'] == 'n' and x['rules_pred'] == 'n']
    v_n_y = [x['scores'] for x in all_results['vanilla'] if x['gold'] == 'n' and x['rules_pred'] == 'y']
    v_y_n = [x['scores'] for x in all_results['vanilla'] if x['gold'] == 'y' and x['rules_pred'] == 'n']
    v_y_y = [x['scores'] for x in all_results['vanilla'] if x['gold'] == 'y' and x['rules_pred'] == 'y']

    v_train = [[x['gold'],x['scores'][:k],success_retrieval(x,k)] for x in all_results_train['vanilla']]
    v = [[x['gold'],x['scores'][:k],success_retrieval(x,k)] for x in all_results['vanilla']]
    v_train = clean_and_pad(v_train)
    v = clean_and_pad(v)


from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ✅ Step 1: 输入数据（使用完整数据替换这里）
data_test = ft
data_train = ft_train
# 🧾 Step 2: 准备特征和标签
X_train = [features for label, features, if_hit in data_train]
y_train = [1 if label == 'y' else 0 for label, features, if_hit in data_train]
X_test = [features for label, features, if_hit in data_test]
y_test = [1 if label == 'y' else 0 for label, features, if_hit in data_test]
if_hit_test = [if_hit for label, features, if_hit in data_test]

print(f'data type: {data_to_load} ft {llm_name} | top_k: {top_k} | eval:{k} ')

# 🪴 Step 4: 训练决策树
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 🔍 Step 5: 分别评估训练集和测试集准确率
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))
print(f'测试集数量：{len(y_test)}')
print(f'训练集数量：{len(y_train)}')
print(f"训练集准确率: {train_acc * 100:.2f}%")
print(f"测试集准确率: {test_acc * 100:.2f}%")
y_test_pred = list(clf.predict(X_test))
# y_pred = ['y' if x == 1 else 'n' for x in list(clf.predict(X_test))]
print_metrics(y_test,y_test_pred)


count_n = sum([1 if y_g == 0 else 0 for y_g in y_test])
count_y = sum([1 if y_g == 1 else 0 for y_g in y_test])

count_n_0_0 = sum([1 if y_g == 0 and not y_p==y_g and not if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_n_0_1 = sum([1 if y_g == 0 and not y_p==y_g and  if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_n_1_0 = sum([1 if y_g == 0 and y_p==y_g and not if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_n_1_1 = sum([1 if y_g == 0 and y_p==y_g and if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])

count_y_0_0 = sum([1 if y_g == 1 and not y_p==y_g and not if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_y_0_1 = sum([1 if y_g == 1 and not y_p==y_g and  if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_y_1_0 = sum([1 if y_g == 1 and y_p==y_g and not if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_y_1_1 = sum([1 if y_g == 1 and y_p==y_g and if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])

count_y_true = sum([1 if y_g == 1 and y_p == 1 and if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_y_false = sum([1 if y_g == 1 and y_p == 0 and if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])


print(count_n_0_0,count_n_0_1,count_n_1_0,count_n_1_1)
print(count_y_0_0,count_y_0_1,count_y_1_0,count_y_1_1)

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



# ✅ Step 1: 输入数据（使用完整数据替换这里）
data_test = v
data_train = v_train
# 🧾 Step 2: 准备特征和标签
X_train = [features for label, features, if_hit in data_train]
y_train = [1 if label == 'y' else 0 for label, features, if_hit in data_train]
X_test = [features for label, features, if_hit in data_test]
y_test = [1 if label == 'y' else 0 for label, features, if_hit in data_test]
if_hit_test = [if_hit for label, features, if_hit in data_test]

print(f'data type: {data_to_load} vanilla {llm_name} | top_k: {top_k} | decision rule: {dr} | eval:{k} ')

# 🪴 Step 4: 训练决策树
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 🔍 Step 5: 分别评估训练集和测试集准确率
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))

print(f'测试集数量：{len(y_test)}')
print(f'训练集数量：{len(y_train)}')
print(f"训练集准确率: {train_acc * 100:.2f}%")
print(f"测试集准确率: {test_acc * 100:.2f}%")
y_test_pred = list(clf.predict(X_test))
# y_pred = ['y' if x == 1 else 'n' for x in list(clf.predict(X_test))]
print_metrics(y_test,y_test_pred)

count_n_0_0 = sum([1 if y_g == 0 and not y_p==y_g and not if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_n_0_1 = sum([1 if y_g == 0 and not y_p==y_g and  if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_n_1_0 = sum([1 if y_g == 0 and y_p==y_g and not if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_n_1_1 = sum([1 if y_g == 0 and y_p==y_g and if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])

count_y_0_0 = sum([1 if y_g == 1 and not y_p==y_g and not if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_y_0_1 = sum([1 if y_g == 1 and not y_p==y_g and  if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_y_1_0 = sum([1 if y_g == 1 and y_p==y_g and not if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_y_1_1 = sum([1 if y_g == 1 and y_p==y_g and if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])

count_y_true = sum([1 if y_g == 1 and y_p == 1 and if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])
count_y_false = sum([1 if y_g == 1 and y_p == 0 and if_hit else 0 for y_g,y_p,if_hit in zip(y_test,y_test_pred,if_hit_test)])


print(count_n_0_0,count_n_0_1,count_n_1_0,count_n_1_1)
print(count_y_0_0,count_y_0_1,count_y_1_0,count_y_1_1)




