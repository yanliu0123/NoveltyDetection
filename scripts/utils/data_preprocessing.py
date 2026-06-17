from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any
import os


from tqdm import tqdm
import pandas as pd
from typing import List, Dict


#----------------------------- 1. START: Preprocessing for alignment task -----------------------------
# 用于生成alignment任务的训练、验证、测试数据集 （怎么感觉跟Step2.的代码重复了？）
def preprocess_data(
    dfs: List[pd.DataFrame], 
    df_dataset: pd.DataFrame, 
    splits: List[str] = ['train', 'val', 'test']
) -> List[List[Dict]]:
    """预处理数据集
    
    Args:
        dfs: 数据集列表 [train_df, val_df, test_df]
        df_dataset: 参考数据集
        splits: 数据集划分名称
    Returns:
        处理后的数据列表
    """
    data = []
    df_ref = df_dataset.drop_duplicates(subset=['paperId'])
    original_cols = ['hypothesis_4omini', 'original1', 'original2']

    for df_name, df in zip(splits, dfs):
        df.fillna('', inplace=True)
        processed = []
        
        for _, row in tqdm(df.iterrows(), desc=f"处理{df_name}数据"):
            try:
                # 提取必要信息
                sample = {
                    'seed': row['paperId'],
                    'original': [row[col] for col in original_cols if row[col] != ''],
                    'keywords': row.get('keywords_4omini'),
                    'publicationDate': pd.to_datetime(row['publicationDate']).strftime('%Y-%m-%d'),
                    'novelty': 'N',
                    'neg_type': row['neg_type'],
                    'bert_score_f1': row['bert_score_f1']
                }
                processed.append(sample)
                
            except Exception as e:
                print(f"处理错误: {str(e)}")
                continue
        
        data.append(processed)
        print(f"{df_name}数据集: 处理{len(processed)}条")
    
    return data
#--------------------------------------------------------------------------------------------------------


#----------------------------- 2. START: Preprocessing for alignment task -----------------------------
# 使用示例：
"""
processed_dfs = generate_alignment_examples(
    dfs=[train_df, val_df, test_df],
    df_names=['train', 'val', 'test'],
    df_ref=reference_df,
    flg_wrong_sample=True
)

# 打印统计信息
for df_name, processed_df in zip(['train', 'val', 'test'], processed_dfs):
    positive = sum(1 for row in processed_df if row['label'] == 1.0)
    negative = sum(1 for row in processed_df if row['label'] == 0.0)
    print(f"\n{df_name} 统计:")
    print(f"总样本数: {len(processed_df)}")
    print(f"正样本数: {positive}")
    print(f"负样本数: {negative}")
"""
def get_latest_date(dates: List[str]) -> str:
    """获取最新日期"""
    return pd.to_datetime(dates).max().strftime('%Y-%m-%d')


def process_single_row(row: pd.Series, df_ref: pd.DataFrame, df_name: str, flg_wrong_sample: bool) -> List[Dict[str, Any]]:
    """处理单行数据"""
    processed_df = []
    given_idea = row['sentence']
    neg_type = row['neg_type']
    novelty = row['novelty']

    # # 根据不同类型处理原始数据和日期
    # if neg_type in ['subset', 'paraphrase']:
    #     original = [row['hypothesis_4omini']]
    #     date = row['publicationDate'].strftime('%Y-%m-%d')
    # elif neg_type == 'combination':
    #     original = [row['original1'], row['original2']]
    #     dates = df_ref[df_ref['hypothesis_4omini'].isin(original)]['publicationDate'].unique()
    #     date = get_latest_date(dates)
    # else:
    #     original = []
    original = [row['hypothesis_4omini']]
    date = row['publicationDate'].strftime('%Y-%m-%d')

    # # 日期处理：加3个月
    # date_obj = datetime.strptime(date, '%Y-%m-%d')
    # future_date = date_obj + relativedelta(months=+3)
    # future_date_str = future_date.strftime('%Y-%m-%d')

    # 获取语料库
    corpus = []
    if df_name in ['val', 'test']:
        corpus = df_ref[df_ref['publicationDate'] <= date]['hypothesis_4omini'].tolist()

    # 生成正样本
    for orig in original:
        processed_df.append({
            'given_idea': given_idea,
            'original': orig,
            'neg_type': neg_type,
            'novelty': novelty,
            'corpus': corpus,
            'publicationDate': date,
            'label': 1.0
        })

    # 生成负样本
    if flg_wrong_sample and df_name in ['val', 'test'] and corpus:
        wrong_samples = [p for p in corpus if p not in original]
        if wrong_samples:  # 确保有可用的负样本
            processed_df.append({
                'given_idea': given_idea,
                'original': wrong_samples[0],
                'neg_type': neg_type,
                'novelty': novelty,
                'corpus': corpus,
                'publicationDate': date,
                'label': 0.0
            })

    return processed_df


def generate_alignment_examples(
    dfs: List[pd.DataFrame], 
    df_names: List[str], 
    df_ref: pd.DataFrame, 
    flg_wrong_sample: bool = True,
    save_path: str = None
) -> List[List[Dict[str, Any]]]:
    """
    生成对齐样本
    
    参数:
        dfs: DataFrame列表
        df_names: DataFrame名称列表
        df_ref: 参考DataFrame
        flg_wrong_sample: 是否生成负样本
    
    返回:
        处理后的数据列表
    """
    processed_dfs = []
    
    # 预处理参考DataFrame
    df_ref['publicationDate'] = pd.to_datetime(df_ref['publicationDate'])
    
    for df, df_name in zip(dfs, df_names):
        print(f"\n处理数据集: {df_name}")
        df['publicationDate'] = pd.to_datetime(df['publicationDate'])
        processed_rows = []
        i
        
        # 使用tqdm显示进度
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {df_name}"):
            try:
                processed_rows.extend(
                    process_single_row(row, df_ref, df_name, flg_wrong_sample)
                )
            except Exception as e:
                print(f"处理行时出错: {str(e)}")
                continue
        
        processed_dfs.append(processed_rows)
        if save_path:
            pd.DataFrame(processed_rows).to_json(os.path.join(save_path, f'{df_name}.json'), orient='records', lines=True)
        print(f"完成 {df_name}: 生成 {len(processed_rows)} 条样本")
    
    return processed_dfs
#--------------------------------------------------------------------------------------------------------


#----------------------------- 3. START: Load all alignment data -----------------------------
import json
def load_all_alignment_data(path_to_root:str,split_names:list):
    datas = []
    for file_name in split_names:
        with open( f'{path_to_root}/alignment_{file_name}.json', 'r') as f:
            data = json.load(f)
        datas.append(data)
    return datas
#--------------------------------------------------------------------------------------------------------


#----------------------------- 4. START: Preprocessing for test novelty checking task -----------------------------
from dateutil.relativedelta import relativedelta
from datetime import datetime

def process_new_neg_sentences(response:str,original:list,def_ref_:pd.DataFrame) -> List[Dict]:
    processed = []
    new_sentences = extract_numbered_sentences(response)
    dates = def_ref_[def_ref_['hypothesis_4omini'].isin(original)]['publicationDate'].unique()
    date = data_preprocessing.get_latest_date(dates)
    new_date = modify_date(date,months=3)
    corpus = def_ref_[def_ref_['publicationDate']<=date]['hypothesis_4omini'].tolist()

    for ns in new_sentences:
        for o in original:
            new_row = {
                'given_idea':ns,
                'original':o,
                'neg_type':'combination',
                'novelty':'N',
                'publicationDate':new_date,
                'corpus':corpus,
                'label':1.0
            }
            processed.append(new_row)
    return processed


def append_to_combination_df(original: list, negs: List[Dict], df_combination_: pd.DataFrame) -> pd.DataFrame:
    """
    将新的组合数据添加到DataFrame中
    
    Args:
        original: 原始句子列表 [original1, original2]
        negs: 负样本列表
        df_combination_: 现有的组合DataFrame
    """
    for i in range(len(negs)):
        new_df = pd.DataFrame({
            'original1': [original[0]],  # 使用列表包装标量值
            'original2': [original[1]],
            'neg_type': [negs[i]['neg_type']],
            'sentence': [negs[i]['given_idea']],
            'publicationDate': [negs[i]['publicationDate']],
            'novelty': [negs[i]['novelty']]
        })
        df_combination_ = pd.concat([df_combination_, new_df], ignore_index=True)  # 添加 ignore_index=True
    
    return df_combination_

def modify_date_of_negative_examples(df_negative_examples:pd.DataFrame,df_ref:pd.DataFrame,months:int=3):
    if df_negative_examples['neg_type'].unique().tolist() == ['paraphrase'] or df_negative_examples['neg_type'].unique().tolist() == ['subset']:
        unique_original = df_negative_examples['original'].unique().tolist()
        for i, original in enumerate(unique_original):
            publicationDate = df_negative_examples[df_negative_examples['hypothesis_4omini']==original]['publicationDate'].unique().tolist()[0]
            date = df_ref[df_ref['hypothesis_4omini']==original]['publicationDate'].unique().tolist()[0]
            new_date = modify_date(date,months=months)
            df_negative_examples.loc[df_negative_examples['original']==original,'publicationDate'] = new_date
            print(f'{i}/{len(unique_original)}: {original} - {publicationDate} -> {new_date}')


    elif df_negative_examples['neg_type'].unique().tolist() == ['combination']:
        unique_originals = df_negative_examples.drop_duplicates(subset=['original1','original2'])['original1','original2'].unique().tolist()
        for i, originals in enumerate(unique_originals):
            dates = df_ref[df_ref['hypothesis_4omini'].isin(originals)]['publicationDate'].unique()
            date = get_latest_date(dates)
            new_date = modify_date(date,months=months)
            index_to_modify = df_negative_examples[df_negative_examples['original1']==originals[0]]&df_negative_examples[df_negative_examples['original2']==originals[1]]   
            df_negative_examples.loc[index_to_modify,'publicationDate'] = new_date
            print(f'{i}/{len(unique_originals)}: {originals} - {date} -> {new_date}')
    return df_negative_examples

def process_combination_examples(df_negative_examples: pd.DataFrame, df_ref: pd.DataFrame, months: int = 3) -> pd.DataFrame:
    """
    处理组合类型的负样本，更新发布日期
    
    Args:
        df_negative_examples: 负样本DataFrame
        df_ref: 参考DataFrame
        months: 要增加的月份数
    Returns:
        更新后的DataFrame
    """
    # 确保是combination类型
    if df_negative_examples['neg_type'].unique().tolist() != ['combination']:
        return df_negative_examples
    
    # 获取唯一的original1和original2组合
    unique_pairs = df_negative_examples[['original1', 'original2']].drop_duplicates()
    
    # 使用tqdm显示进度
    for idx, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="处理组合样本"):
        try:
            # 获取当前组合
            orig1, orig2 = row['original1'], row['original2']
            
            # 获取相关日期
            dates = df_ref[df_ref['hypothesis_4omini'].isin([orig1, orig2])]['publicationDate']
            if not dates.empty:
                # 获取最新日期并修改
                latest_date = get_latest_date(dates)
                new_date = modify_date(latest_date, months=months)
                
                # 更新符合条件的行
                mask = (df_negative_examples['original1'] == orig1) & \
                      (df_negative_examples['original2'] == orig2)
                df_negative_examples.loc[mask, 'publicationDate'] = new_date
                
                print(f"更新组合 [{orig1}, {orig2}]: {latest_date} -> {new_date}")
                
        except Exception as e:
            print(f"处理组合时出错 [{orig1}, {orig2}]: {str(e)}")
            continue
    
    return df_negative_examples

# 辅助函数
def get_latest_date(dates: pd.Series) -> str:
    """获取最新日期"""
    return pd.to_datetime(dates).max().strftime('%Y-%m-%d')

def modify_date(date: str, months: int = 3) -> str:
    """修改日期"""
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    new_date = date_obj + relativedelta(months=months)
    return new_date.strftime('%Y-%m-%d')

    # 使用示例：
    """
    # 导入必要的库
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    from tqdm import tqdm
    import pandas as pd

    # 使用函数
    df_negative_examples = process_combination_examples(
        df_negative_examples=your_df,
        df_ref=reference_df,
        months=3
    )
    """

def create_expanded_combination_df(df_negative_combination: pd.DataFrame) -> pd.DataFrame:
    """
    将combination数据展开，每行拆分为两行
    
    Args:
        df_negative_combination: 原始DataFrame
    Returns:
        展开后的新DataFrame
    """
    # 创建第一组数据（使用original1）
    df1 = pd.DataFrame({
        'paperId': df_negative_combination['paperId'],
        'original': df_negative_combination['original1'],
        'publicationDate': df_negative_combination['publicationDate'],
        'sentence': df_negative_combination['sentence'],
        'neg_type': df_negative_combination['neg_type'],
        'novelty': df_negative_combination['novelty'],
        'title': df_negative_combination['title']
    })

    # 创建第二组数据（使用original2）
    df2 = pd.DataFrame({
        'paperId': df_negative_combination['paperId'],
        'original': df_negative_combination['original2'],
        'publicationDate': df_negative_combination['publicationDate'],
        'sentence': df_negative_combination['sentence'],
        'neg_type': df_negative_combination['neg_type'],
        'novelty': df_negative_combination['novelty'],
        'title': df_negative_combination['title']
    })

    # 合并两个DataFrame
    df_combination = pd.concat([df1, df2], ignore_index=True)
    
    print(f"原始数据行数: {len(df_negative_combination)}")
    print(f"展开后行数: {len(df_combination)}")
    
    return df_combination
    '''
    # 使用示例：
    df_negative_combination = pd.read_csv('negative_combination.csv')
    df_expanded = create_expanded_combination_df(df_negative_combination)
    df_expanded.to_csv('expanded_combination.csv', index=False)

    # 使用示例：
    df_combination = create_expanded_combination_df(df_negative_combination)

    # 显示结果
    print("\n展开后的前几行:")
    print(df_combination.head())

    # 验证数据
    print("\n数据验证:")
    print(f"唯一paperId数: {df_combination['paperId'].nunique()}")
    print(f"每个paperId的行数: \n{df_combination.groupby('paperId').size().value_counts()}")
    '''
#--------------------------------------------------------------------------------------------------------