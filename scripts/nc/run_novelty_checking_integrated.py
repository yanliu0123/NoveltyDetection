# === Imports ===
import os
import json
import random
import warnings
from typing import List, Dict, Tuple
from regex import R
from tqdm import tqdm
from deepseek_parallel import get_answers
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# === Warnings Configuration ===
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
price = {
'4omini': {
    'prompt': 0.4/1000000.0,
    'completion': 1.6/1000000.0
    },
    '4o': {
        'prompt': 2.5/1000000.0,
        'completion': 10/1000000.0
    },
    'deepseek': {
        'prompt': 2/1000000.0,
        'completion': 8/1000000.0
    }
}
# === LLM Initialization Functions ===
def initialize_llama_model(server: str = 'duy') -> Tuple:
    model_dir = f'/home/{server}/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype='auto', device_map='auto')
    # if tokenizer.pad_token is None:
    tokenizer.eos_token_id = 128001
    tokenizer.pad_token = tokenizer.eos_token  # Set to eos_token if pad_token is not defined
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id =  tokenizer.eos_token_id  
    return tokenizer, model

def initialise_qwen():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M", torch_dtype='auto', device_map="auto")
    return tokenizer, model

# === Query Functions ===
def query_llama(prompt: str, model, tokenizer) -> str:
    system_prompt = 'You are an expert in scientific novelty assessment. You will rate novelty on a 0-1 scale.'
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024, temperature=0.00001, top_p=0.9,do_sample=False, attention_mask=model_inputs.attention_mask)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.split('assistant\n\n')[-1]

def query_qwen(prompt,model,tokenizer,system_prompt=None):
    if system_prompt is None:
        system_prompt = 'You are an AI assistant specializing in scientific novelty assessment. Your task is to evaluate the originality of a given research hypothesis or idea by comparing it with existing studies. Your response should provide a novelty score between 0 and 1, highlight similarities with prior research, and a confidence score between 0 and 1.'
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=2000,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Optionally extract just the assistant's part
    # assistant_reply = response.split("<|im_start|>assistant\n")[-1]
    # print(assistant_reply.strip())
    return response.split('\n\n\n\n```python\n')[-1].lower()[-90:]

from openai import OpenAI
from openai import AzureOpenAI
def query_gpt4o_mini(prompt, system_prompt=None, api_key='4a191e8939e14ac29a6fd369de581b24'):
    if not system_prompt:
        system_prompt = 'You are an AI assistant specializing in scientific novelty assessment. Your task is to evaluate the originality of a given research hypothesis or idea by comparing it with existing studies. Your response should provide a novelty score list and a final decision (Y or N), without any other explanation.'

    client = AzureOpenAI(
        azure_endpoint="https://declaregpt4.openai.azure.com/",
        api_key=api_key,
        api_version="2024-08-01-preview",
    )

    message_text = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    completion = client.chat.completions.create(
        model='GPT4O-mini',
        messages=message_text,
        temperature=0.0,
        max_tokens=300
    )

    response = completion.choices[0].message.content
    token_usage = {
        'prompt_tokens': completion.usage.prompt_tokens,
        'completion_tokens': completion.usage.completion_tokens,
        'total_tokens': completion.usage.total_tokens
    }
    return response, token_usage

def query_gpt4o(prompt, system_prompt=None, api_key='4a191e8939e14ac29a6fd369de581b24'):
    if not system_prompt:
        system_prompt = 'You are an AI assistant specializing in scientific novelty assessment. Your task is to evaluate the originality of a given research hypothesis or idea by comparing it with existing studies. Your response should provide a novelty score list and a final decision (Y or N), without any other explanation.'
 
    client = AzureOpenAI(
    azure_endpoint = "https://declaregpt4.openai.azure.com/", 
    # azure_endpoint = "https://dalle-declare.openai.azure.com/openai/deployments/o1-mini/chat/completions?api-version=2024-08-01-preview",  # 4o-mini

    # api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_key=api_key,
    api_version="2024-08-01-preview",
    )

    message_text = [{"role":"system","content":system_prompt},
                    {"role":"user","content":prompt}
                    ]

    completion = client.chat.completions.create(
    model='GPT4O', # model = "deployment_name"
    messages = message_text,
    temperature=0.0,
    max_tokens=300,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
    )
    response = completion.choices[0].message.content
    token_usage = {
        'prompt_tokens': completion.usage.prompt_tokens,
        'completion_tokens': completion.usage.completion_tokens,
        'total_tokens': completion.usage.total_tokens
    }
    return response,token_usage

def query_deepseek(prompt, system_prompt=None, api_key='sk-3c8c039115354c2da2208586cd1b2b55'):
    if not system_prompt:
        system_prompt = 'You are an AI assistant specializing in scientific novelty assessment.'

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0.0
    )
    reply = response.choices[0].message.content
    token_usage = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }
    return reply, token_usage

def query_llm_model(llm_name, prompt, model=None, tokenizer=None):
    if llm_name == 'llama':
        response = query_llama(prompt, model, tokenizer)
        return response, {}
    elif llm_name == 'qwen':
        response = query_qwen(prompt, model, tokenizer)
        return response, {}
    elif llm_name == '4omini':
        return query_gpt4o_mini(prompt)
    elif llm_name == '4o':
        return query_gpt4o(prompt)
    elif llm_name == 'deepseek':
        return query_deepseek(prompt)
    else:
        raise ValueError(f"Unsupported LLM: {llm_name}")


# === Prompt Composition ===
def compose_prompt_novelty_assessment_new(query, retrieved_results, top_k=5,flag_use_retrieved_results=True):
   retrieved_results = '\n'.join([f'{i+1}. {result}' for i, result in enumerate(retrieved_results[:top_k])])
   prompt = f"""
**Task Overview:**

You are given a *"new research idea"* and a list of *"existing research ideas."* Your task is to compare the new idea **against each existing idea** and assign a **novelty score**, followed by a final decision on whether the idea is **novel or not**.

---

### Clarified Scoring Criteria:

Use the following **novelty scoring rubric** for each comparison:

Scoring Guidelines:  
0.0 – No Novelty (Identical / Reworded)
The new idea is a direct copy or a rephrased version of an existing idea.
Shares the same claims, findings, information or logic.
Example: Paraphrasing an existing idea without introducing any change.

0.3 (exclusive) – Low Novelty (Subset of Existing Idea)
The new idea is a strict subset of an existing idea.
Removes a condition, claim, or component, but otherwise shares the same logic and goal.
No new dimension or direction added.
Example: Taking just “Claim A” from “Claim A + Claim B”.

0.5 (exclusive) – Moderate Novelty (large partial overlap)
The new idea shares a substantial portion (around 50%) of the ideas or claims with an existing idea.
It introduces some new elements, such as a different combination of claims or modified emphasis, but it’s not entirely new.
It’s not a subset or superset, but has a large information intersection.
Example: “Claim A + Claim C” compared to existing “Claim A + Claim B”.

0.7 (exclusive) – High Novelty (small partial overlap )
The new idea has minor similarities (e.g., method or framing), but differs in research focus, target, or core claim.
It applies known ideas in a new context or formulates a distinct question, showing clear divergence from existing ideas.
Example: Using a social theory from marketing in the context of education.

1.0 – Very High Novelty (Distinct Research Direction)
The idea is entirely distinct in structure, claim, scope, and research objective.
Only minor thematic or terminological similarity, if any.
Example: Proposing a brand new framework, theory, or dataset with no precedent in existing ideas.


---
### Evaluation Instructions:
For each comparison between the new idea and an existing idea:
1. Assess Overlap
   Examine the conceptual and structural overlap between the new idea and the existing idea.

2. Assign a Novelty Score
   Use the Novelty Scoring Rubric to assign a score between 0.0 and 1.0, based on the degree of similarity or difference.

3. Repeat for All Comparisons
   Perform this scoring for each existing idea in the set.

4. Apply Final Decision Rule (see below).

---

### Final Decision Rule:

* If the minimum score across all comparisons is in the low to moderate range (0.0-0.5), the idea is not novel → return N.
* If the minimum score is in the high range (0.5-1.0), the idea is novel → return Y.
---

```
### Output Format:
[List of scores like: [0.3, 0.5, 0.3, 0.7, 1.0]]
Final Decision: Y / N 
```
Now, compare the given research idea with each of the existing ideas. For each comparison, assign a novelty score using the rubric above and no need to justify the score. Then apply the final decision rule to determine if the new idea is novel.Please return only a Python-style list of the novelty scores for each comparison and the final decision, no further explanation.


Given Idea:
{query}

"""
   if flag_use_retrieved_results:
   
      prompt += f"""
       Existing Ideas
        {retrieved_results}
        """
   return prompt

def compose_prompt_novelty_assessment_zl(query,displine):
    
    prompts = f"You are assisting {displine} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {displine} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be novel enough, which means it should not have been proposed by any existing literature before. \nPlease directly answer this question. The hypothesis is: \n{query}\n"
    prompts+= "Please give a response to the initial question on determining whether the research hypothesis is novel enough. If you think it is novel enough, leave your response as 'Y'. Otherwise, leave your response as 'N'. Your response should be only 'Y' or 'N'."
    return prompts
# === Response Parsing ===
def parse_response_and_decision(response: str):
    response = response.lower()
    try:
        scores = response.split('[')[1].split(']')[0].split(',')
        scores = [float(score.strip()) for score in scores]
        # 自动判断
        min_score = min(scores) if scores else 0
        decision = 'y' if min_score >= 0.5 else 'n'
        return scores, decision
    except:
        return [], 'n'
    
def rules(scores):
    if 0.0 in scores or 0.3 in scores:
        return 'n'
    elif scores.count(0.5)>=2:
        return 'n'
    else:
        return 'y' 
    
# === Evaluation Loop ===
def run_evaluation_deepseek(nc_test_data: List[Dict], model, tokenizer, total_token_usage, total_price, top_k: int, save_path: str, llm_name: str,data_type:str,domain:str,mode:str):
    y_true, y_pred, logs = [], [], []
    count_y_y, count_n_n, count_y_n, count_n_y = 0, 0, 0, 0
    prompts = []
    for i, item in enumerate(nc_test_data):
        query = item['given_idea']
        retrieved = [x['sentence'] for x in item['top_k'][:top_k]]
        if mode == 'zonglin':
            displine = 'natural language processing' if domain == 'acl' else 'marketing'
            prompt = compose_prompt_novelty_assessment_zl(query,displine)

        else:
            prompt = compose_prompt_novelty_assessment_new(query, retrieved, top_k)
        prompts.append(prompt)

    # print(f'prompts: {prompts}')
    answers = get_answers(prompts)
    

    for i, item in tqdm(enumerate(nc_test_data), total=len(nc_test_data), desc=f"Running {llm_name} top_k={top_k} {data_type} {domain} {mode}"):
        query = item['given_idea']
        retrieved = [x['sentence'] for x in item['top_k'][:top_k]]
        gold = item['novelty'].strip().lower()



        response,token_usage = answers[i],False #query_llm_model(llm_name, prompt, model, tokenizer)
        if token_usage:
            total_token_usage['prompt_tokens'] += token_usage['prompt_tokens']
            total_token_usage['completion_tokens'] += token_usage['completion_tokens']
            total_token_usage['total_tokens'] += token_usage['total_tokens']
            total_price['prompt_price'] += total_token_usage['prompt_tokens'] * price[llm_name]['prompt']
            total_price['completion_price'] += total_token_usage['completion_tokens'] * price[llm_name]['completion']
            total_price['total_price'] = total_price['prompt_price'] + total_price['completion_price']
            print(f" Price: {total_price['total_price']} | input {token_usage['prompt_tokens']} tokens | output {token_usage['completion_tokens']} tokens")
        scores, decision_rule = parse_response_and_decision(response)
        # if mode == 'auto':
        # print(f"response: {response}")
        decision_rule = rules(scores)
        decision = 'y' if 'y' in response.lower()[-1:] else 'n'

        if mode == 'zonglin':
            decision_rule = decision
        # print(f"rules Pred: {decision}")
        if gold == 'y' and decision_rule == 'y' :
            count_y_y += 1
        elif gold == 'n' and decision_rule == 'n':
            count_n_n += 1
        elif gold == 'n' and decision_rule == 'y':
            count_n_y += 1
        elif gold == 'y' and decision_rule == 'n':
            count_y_n += 1

        print(f"{i+1}/{len(nc_test_data)} | GT: {gold} | llm Pred: {decision} | rules Pred (): {decision_rule} | scores: {scores} | count_y_y: {count_y_y} | count_n_n: {count_n_n} | count_y_n: {count_y_n} | count_n_y: {count_n_y}\n")
        
        y_true.append(gold)
        if mode == 'auto':
            y_pred.append(decision)
        elif mode == 'zonglin':
            y_pred.append(decision)
        else:
            y_pred.append(decision_rule)
        logs.append({
            'id': item['id'],
            'query': query,
            'gold': gold,
            'llm_pred': decision,
            'rules_pred': decision_rule,
            'is_correct': decision == gold,
            'scores': scores,
            'response': response,
            'top_k': item['top_k'][:top_k]
        })
        if (i+1)%1 == 0:
            with open(save_path, 'w+') as f:
                for log in logs:
                    f.write(json.dumps(log, ensure_ascii=False) + '\n')
            # logs = []


    with open(save_path, 'w+') as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')

    print_metrics(y_true, y_pred)

def run_evaluation_others(nc_test_data: List[Dict], model, tokenizer, total_token_usage, total_price, top_k: int, save_path: str, llm_name: str,data_type:str,domain:str,mode:str):
    y_true, y_pred, logs = [], [], []
    count_y_y, count_n_n, count_y_n, count_n_y = 0, 0, 0, 0
    # prompts = []

    for i, item in tqdm(enumerate(nc_test_data), total=len(nc_test_data), desc=f"Running {llm_name} top_k={top_k} {data_type} {domain} {mode}"):
        query = item['given_idea']
        retrieved = [x['sentence'] for x in item['top_k'][:top_k]]
        gold = item['novelty'].strip().lower()

        # prompt = compose_prompt_novelty_assessment_new(query, retrieved, top_k)
        if mode == 'zonglin':
            displine = 'natural language processing' if domain == 'acl' else 'marketing'
            prompt = compose_prompt_novelty_assessment_zl(query,displine)
        else:
            prompt = compose_prompt_novelty_assessment_new(query, retrieved, top_k)


        response,token_usage = query_llm_model(llm_name, prompt, model, tokenizer)

        if token_usage:
            total_token_usage['prompt_tokens'] += token_usage['prompt_tokens']
            total_token_usage['completion_tokens'] += token_usage['completion_tokens']
            total_token_usage['total_tokens'] += token_usage['total_tokens']
            total_price['prompt_price'] += total_token_usage['prompt_tokens'] * price[llm_name]['prompt']
            total_price['completion_price'] += total_token_usage['completion_tokens'] * price[llm_name]['completion']
            total_price['total_price'] = total_price['prompt_price'] + total_price['completion_price']
            print(f" Price: {total_price['total_price']} | input {token_usage['prompt_tokens']} tokens | output {token_usage['completion_tokens']} tokens")
        scores, decision_rule = parse_response_and_decision(response)
        print(f'response({type(response)},len:{len(response)}): {response}')
        # if mode == 'auto':
        # print(f"response: {response}")
        decision_rule = rules(scores)
        decision = 'y' if 'y' in response.lower()[-1:] else 'n'
        # print(f"rules Pred: {decision}")
        if mode == 'zonglin':
            decision_rule = decision
        if gold == 'y' and decision_rule == 'y' :
            count_y_y += 1
        elif gold == 'n' and decision_rule == 'n':
            count_n_n += 1
        elif gold == 'n' and decision_rule == 'y':
            count_n_y += 1
        elif gold == 'y' and decision_rule == 'n':
            count_y_n += 1

        print(f"{i+1}/{len(nc_test_data)} | GT: {gold} | llm Pred: {decision} | rules Pred (): {decision_rule} | scores: {scores} | count_y_y: {count_y_y} | count_n_n: {count_n_n} | count_y_n: {count_y_n} | count_n_y: {count_n_y}\n")
        
        y_true.append(gold)
        if mode == 'auto':
            y_pred.append(decision)
        elif mode == 'zonglin':
            y_pred.append(decision)
        else:
            y_pred.append(decision_rule)
        logs.append({
            'id': item['id'],
            'query': query,
            'gold': gold,
            'llm_pred': decision,
            'rules_pred': decision_rule,
            'is_correct': decision == gold,
            'scores': scores,
            'response': response,
            'top_k': item['top_k'][:top_k]
        })
        if (i+1)%10 == 0:
            with open(save_path, 'w+') as f:
                for log in logs:
                    f.write(json.dumps(log, ensure_ascii=False) + '\n')
            # logs = []


    with open(save_path, 'w+') as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')

    print_metrics(y_true, y_pred)

# === Metrics ===
def print_metrics(y_true: List[str], y_pred: List[str]):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label="y")
    print("\n📊 Evaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

# === Entry ===
if __name__ == '__main__':
    domain = 'acl'
    embedding_model_name = 'BGE'
    data_type_list = ['vanilla']
    top_k_list = [20]
    llm_names = ['deepseek']  # Extend as needed
    mode = 'manual'
    data_sufixs = ['100_for_test','100_for_train']

    total_token_usage = {'prompt_tokens':0, 'completion_tokens':0, 'total_tokens':0}
    total_price = {'prompt_price':0, 'completion_price':0, 'total_price':0}

    for llm_name in llm_names:
        tokenizer, model = (None, None)
        if llm_name in ['llama', 'qwen']:
            tokenizer, model = initialize_llama_model() if llm_name == 'llama' else initialise_qwen()

        for data_type in data_type_list:
            for data_sufix in data_sufixs:
                data_path = f'/home/duy/ly/novelty_checker/notebooks/clean_version/constrastive_model/0518/{domain}/prepared_for_nc/nc_results_{domain}_{embedding_model_name}_{data_type}_{data_sufix}.json'
                with open(data_path, 'r') as f:
                    nc_test_data = json.load(f)
                    print('!!',len(nc_test_data))
                    # random.shuffle(nc_test_data)

                for top_k in top_k_list:
                    print(f'Running {domain} {embedding_model_name} {data_type} {llm_name} {top_k} {mode} {data_sufix}')

                    save_path = f'/data/duy/ly/{domain}/versions/20250512/nc_results/0520/{llm_name}/nc_test_{domain}_{embedding_model_name}_{data_type}_{llm_name}_top_{top_k}_{mode}_{data_sufix}_zonglin.json'
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    if llm_name == 'deepseek':
                        run_evaluation_deepseek(nc_test_data, model, tokenizer, total_token_usage, total_price, top_k, save_path, llm_name,data_type,domain,mode)
                    else:
                        run_evaluation_others(nc_test_data, model, tokenizer, total_token_usage, total_price, top_k, save_path, llm_name,data_type,domain,mode)

