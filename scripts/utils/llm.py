import torch
import re
import numpy as np
import pandas as pd
from transformers import  pipeline
# from zhipuai import ZhipuAI
from openai import OpenAI
from openai import AzureOpenAI
import openai

import requests

system_prompt_marketing = 'You are a social science expert searching for new research topics to explore. To accomplish this, you need to conduct a comprehensive literature review to understand the current research landscape, including the challenges being addressed, the hypotheses being tested, the methodologies employed, and the specific conditions under which key findings are obtained.'
system_prompt_nlp = 'You are a computer science expert searching for new research ideas. To accomplish this, you need to conduct a comprehensive literature review to understand the current research landscape, including the research tasks, the methods used on the tasks, and the specific conditions under which key findings are obtained.'

kimi_api_key = 'sk-ernFkIOak7n4VBecvLutwdrHm92Pa6SmjmVE1lLGrGsnEOwS'
gpt4_api_key = '4a191e8939e14ac29a6fd369de581b24'
glm_api_key = '531a077e1b5574eac3cbc22bbdce219e.wt63KJbgSy4AHbsc'
gpt_o1_mini_api_key = '6e83c9da40724b3bbb02b74ca8c30d5b'
gpt4o_mini_api_key_ = '4a191e8939e14ac29a6fd369de581b24'


# ===================== Prompt Marketing ========================
def compose_prompt_extract_paper_details_marketing(abstract,predicates=None,entities=None,title = None):
    
    prompt = ''
    # flag_include_predicates = False
    # flag_include_entities = False
    flag_include_task_description = True
    flag_include_example = True
    flag_include_background = True
    # flag_include_title =

    if flag_include_background:
        prompt += '''In social science research, a hypothesis is typically a clear, specific, and testable statement predicting the relationship between an independent variable and a dependent variable. It often outlines how one variable influences another and to what degree. The hypothesis is central to the research, as it guides the study's aim of proving or disproving this relationship. In many cases, the hypothesis reflects the main conclusion or finding of the research. \nKeywords or key phrases are essential terms capturing the study's core concepts, variables, and major themes, and should reflect the primary focus areas. They should neither be too specific nor too general, avoiding keyword over-expansion while effectively indexing the research.'''
    
    if flag_include_task_description:
        prompt += '''Your task is to extract the hypothesis and main keywords from each title and abstract pair of a research paper. Then, build a knowledge graph by identifying relationships between these keywords, using predefined predicates wherever possible. Construct triples in the form of (subject-predicate-object) and ensure that predicates and entities are balanced in specificity. Avoid generating excessive or overly granular keywords and predicates. If no suitable relationship exists, you can propose a new predicate with a brief description and example. Note that the predicates should not be overly case-specific to avoid predicate explosion.  If you cannot find any hypothesis in the abstract, you can extract the findings or conclusions of the contributions of the study as the hypothesis. If the input abstract is not meaningful, you can output 'None'.'''

    if predicates:
        prompt += '''\n\nThe existing predicates and their descriptions are(each in the format, predicate_name, predicate_description, predicate_usage_example):\n\t'''
        prompt += predicates

    if entities:
        prompt += '\n\nThe existing entities are:\n\t'
        prompt += entities


    prompt += '''\n\nGiven time constraints, you only need to review the title and abstract of research papers published in top-tier journals. \nYour task is to extract the hypothesis, keywords and relationships between the keywords from each title and abstract pair. The keywords must include the variables, and the hypothesis should be comprehensive, including key details or side hypotheses if present. If no hypothesis is found, you can extract the findings or conclusions of the contributions of the study as the hypothesis.'''
    
    prompt += '''\n\nBelow is the target article:```\n'''
    if title is not None:
        prompt += f'\tTitle: {title}\n\n'
    prompt += f'\tAbstract: {abstract}'
    prompt += '''```\nAfter reviewing the abstract, please extract the main hypothesis and relevant keywords using the following format (only the hypothesis and keywords, no additional formatting, no additional text to explain your answer, leave it blank if no new predicate or triples are extracted, remember to replace the keywords1, predicates, ... with the keywords and predicates in the abstract; keywords or key phrases from the given abstract are to serve as nodes for a knowledge graph. Ensure that the extracted terms are in their most simplified form, removing redundancies and maintaining a reasonable number of distinct nodes.Keywords, Triples, and New_predicates should be a literal list, each extraction are end with a #):
    Hypothesis: [Extracted hypothesis, including details if present, put value to be 'None' if no hypothesis is found]#
    Keywords: ['keyword1', 'keyword2', 'keyword3', ..., leave it blank if no new predicate is extracted]#
    Triples:  [('keyword1', 'predicate1', 'keyword2'),  ('keyword1', 'predicate2', 'keyword2'),  ('keyword1', 'predicate3', 'keyword2'), ..., leave it blank if no new predicate is extracted]#
    New_predicates: [('predicate', 'description', 'example'), ('predicate', 'description', 'example'), ('predicate', 'description', 'example') , ...,leave it blank if no new predicate is extracted, use underscore to replace space in predicate]#'''
    if flag_include_example:  
        prompt += '''\n\nOne example of the extraction is (assume predicate "heighten" does not exist in the predicates provides):\n
    Hypothesis: Viewing a visually depicted product that facilitates embodied mental simulation leads to heightened purchase intentions, with perceptual resources for mental simulation attenuating this effect, and for negatively valenced products, it decreases purchase intentions.#
    Keywords: ['mental simulation', 'purchase intention', 'consumer behavior']#
    Triples: [('mental simulation', 'heightened', 'purchase intention'), ('mental simulation', 'heightened', 'consumer behavior')]#
    New_predicates: [('heighten', 'A heightened B', '(A,heighten,B)')]# 
    '''
    


    return prompt

def compose_prompt_extract_paper_details_nlp(abstract,predicates=None,entities=None,title = None):
    prompt = ''
    # flag_include_predicates = False
    # flag_include_entities = False
    flag_include_task_description = True
    flag_include_example = False
    flag_include_background = True
    # flag_include_title =

    if flag_include_background:
        prompt += '''
A research hypothesis in a computer science paper is a testable statement or prediction about a specific phenomenon, system, or algorithm that the research aims to investigate or validate. It serves as the foundation of the study, guiding the research design, experiments, and analysis. The hypothesis should be clearly articulated and typically stems from prior knowledge, gaps in existing literature, or a novel idea the authors aim to explore. A good hypothesis should be specific, testable, context-dependent, and relevant to the research problem. When extracting the hypothesis, ensure it begins directly with the subject of the statement (e.g., "Optimistic posterior sampling algorithm for reinforcement learning (OPSRL)...") without introductory phrases such as "The proposed" or "We propose." Focus on presenting the hypothesis succinctly and directly.
Examples of Research Hypotheses in Computer Science:
Algorithm Development: "METHODS will perform better in terms of computational efficiency and accuracy compared to existing algorithms for large-scale data sorting."
Machine Learning: "Incorporating domain-specific embeddings in the neural network architecture will significantly improve its performance on task X."

Keywords are concise terms or phrases that represent the core content and focus areas of a computer science paper. They help in indexing the paper for search engines, databases, and research repositories, making it easier for researchers to discover your work. They should neither be too specific nor too general, avoiding keyword over-expansion while effectively indexing the research.
Examples of Research Hypotheses in Computer Science:
Algorithm Development: "The proposed algorithm will perform better in terms of computational efficiency and accuracy compared to existing algorithms for large-scale data sorting."
Machine Learning: "Incorporating domain-specific embeddings in the neural network architecture will significantly improve its performance on task X."

'''
    
    if flag_include_task_description:
        prompt += '''Your task is to extract the hypothesis and main keywords from each title and abstract pair of a research paper. Then, build a knowledge graph by identifying relationships between these keywords, using predefined predicates wherever possible. Construct triples in the form of (subject-predicate-object) and ensure that predicates and entities are balanced in specificity. Avoid generating excessive or overly granular keywords and predicates. If no suitable relationship exists, you can propose a new predicate with a brief description and example. Note that the predicates should not be overly case-specific to avoid predicate explosion.  If you cannot find any hypothesis in the abstract, you can extract the findings or conclusions of the contributions of the study as the hypothesis. If the input abstract is not meaningful, you can output 'None'.'''

    if predicates:
        prompt += '''\n\nThe existing predicates and their descriptions are(each in the format, predicate_name, predicate_description, predicate_usage_example):\n\t'''
        prompt += predicates

    if entities:
        prompt += '\n\nThe existing entities are:\n\t'
        prompt += entities


    # prompt += '''\n\nGiven time constraints, you only need to review the title and abstract of research papers published in top-tier journals. \nYour task is to extract the hypothesis, keywords and relationships between the keywords from each title and abstract pair. The keywords must include the variables, and the hypothesis should be comprehensive, including key details or side hypotheses if present.'''
    
    prompt += '''\n\nBelow is the target article:```\n'''
    if title is not None:
        prompt += f'\tTitle: {title}\n\n'
    prompt += f'\tAbstract: {abstract}'
    prompt += '''```\nAfter reviewing the abstract, please extract the main hypothesis and relevant keywords using the following format (only the hypothesis and keywords, no additional formatting, no additional text to explain your answer, leave it blank if no new predicate or triples are extracted, remember to replace the keywords1, predicates, ... with the keywords and predicates in the abstract; keywords or key phrases from the given abstract are to serve as nodes for a knowledge graph. Ensure that the extracted terms are in their most simplified form, removing redundancies and maintaining a reasonable number of distinct nodes.Keywords, Triples, and New_predicates should be a literal list, each extraction are end with a #):
    Hypothesis: [Extracted hypothesis, including details if present, use full name rather than abbreviation, put value to be 'None' if no hypothesis is found]#
    Keywords: ['keyword1', 'keyword2', 'keyword3', ..., leave it blank if no new predicate is extracted]#
    Triples:  [('keyword1', 'predicate1', 'keyword2'),  ('keyword1', 'predicate2', 'keyword2'),  ('keyword1', 'predicate3', 'keyword2'), ..., leave it blank if no new predicate is extracted]#
    New_predicates: [('predicate', 'description', 'example'), ('predicate', 'description', 'example'), ('predicate', 'description', 'example') , ...,leave it blank if no new predicate is extracted, use underscore to replace space in predicate]#'''
    if flag_include_example:  
        prompt += '''\n\nOne example of the extraction is (assume predicate "heighten" does not exist in the predicates provides):\n
    Hypothesis: Viewing a visually depicted product that facilitates embodied mental simulation leads to heightened purchase intentions, with perceptual resources for mental simulation attenuating this effect, and for negatively valenced products, it decreases purchase intentions.#
    Keywords: ['mental simulation', 'purchase intention', 'consumer behavior']#
    Triples: [('mental simulation', 'heightened', 'purchase intention'), ('mental simulation', 'heightened', 'consumer behavior')]#
    New_predicates: [('heighten', 'A heightened B', '(A,heighten,B)')]# 
    '''

    return prompt

# ===================== LLAMA ========================
def process_llama_response(response, model='gpt-4o'):
    # Regular expression pattern to match Hypothesis and Keywords
    text = response[0].split('assistant\n\n')[1]

    hypothesis_pattern = r"Hypothesis: (.*)#"
    keywords_pattern = r"Keywords: (.*)#"
    triples_pattern = r"Triples: (.*)#"
    new_predicates_pattern = r"New_predicates: (.*)#"
    
    # Extract the hypothesis
    hypothesis_match = re.search(hypothesis_pattern, text)
    hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else None

    # Extract the keywords
    keywords_match = re.search(keywords_pattern, text)
    keywords = keywords_match.group(1).strip() if keywords_match else None

    triples_match = re.search(triples_pattern, text)
    triples = triples_match.group(1).strip() if triples_match else None

    new_predicates_match = re.search(new_predicates_pattern, text)
    new_predicates = new_predicates_match.group(1).strip() if new_predicates_match else None
    
    return hypothesis, keywords, triples, new_predicates

  # Function to check if hypothesis is valid

def extract_hypothesis_using_llama3(prompt,model,tokenizer,system_prompt=system_prompt_marketing,):
    messages=[{'role':'system','content':system_prompt},
    {'role': 'user','content': prompt}]

    text =tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
    model_inputs=tokenizer([text],return_tensors="pt").to('cuda')
    # model_inputs=tokenizer([text],return_tensors="pt").to('cpu')


    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
        )
        
            # 对输出进行解码
    response=tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    hypothesis, keywords, triples, new_predicates = process_llama_response(response)
    return hypothesis, keywords, triples, new_predicates


# ===================== PHI3 ========================
def process_phi3_response(response):
    # Regular expression pattern to match Hypothesis and Keywords
    text = response[0]['generated_text']

    hypothesis_pattern = r"Hypothesis: (.*)#"
    keywords_pattern = r"Keywords: (.*)#"
    triples_pattern = r"Triples: (.*)#"
    new_predicates_pattern = r"New_predicates: (.*)#"
    
    # Extract the hypothesis
    hypothesis_match = re.search(hypothesis_pattern, text)
    hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else None
    hypothesis = hypothesis.replace('[', '').replace(']', '') if hypothesis else hypothesis
    # Extract the keywords
    keywords_match = re.search(keywords_pattern, text)
    keywords = keywords_match.group(1).strip() if keywords_match else None

    triples_match = re.search(triples_pattern, text)
    triples = triples_match.group(1).strip() if triples_match else None

    new_predicates_match = re.search(new_predicates_pattern, text)
    new_predicates = new_predicates_match.group(1).strip() if new_predicates_match else None
    
    return hypothesis, keywords, triples, new_predicates

def extract_hypothesis_using_phi3(prompt,model,tokenizer):
    messages = [
        {"role": "system", "content": "You are a social science expert searching for new research topics to explore. To accomplish this, you need to conduct a comprehensive literature review to understand the current research landscape, including the challenges being addressed, the hypotheses being tested, the methodologies employed, and the specific conditions under which key findings are obtained."},
        {"role": "user", "content": prompt}]

    
    # Access the underlying model for the pipeline
    model_for_pipeline = model.module if isinstance(model, torch.nn.DataParallel) else model

    pipe = pipeline(
        "text-generation",
        model=model_for_pipeline,
        tokenizer=tokenizer,
    )
    
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    
    output = pipe(messages, **generation_args)
    hypothesis, keywords, triples, new_predicates = process_phi3_response(output)
    return hypothesis, keywords, triples, new_predicates


# ===================== GPT4 ========================
def process_gpt_response(completion,df_cost_stat):
    # Using non-greedy match with .*? to avoid potential issues with multiple sections
    response = completion.choices[0].message.content
    pattern_hypothesis = r"Hypothesis: (.*?)\#"
    pattern_keywords = r"Keywords: (.*?)\#"
    pattern_triples = r"Triples: (.*?)\#"
    pattern_new_predicates = r"New_predicates: (.*?)\#"
    
    hypothesis_match = re.search(pattern_hypothesis, response, re.DOTALL)
    keywords_match = re.search(pattern_keywords, response, re.DOTALL)
    triples_match = re.search(pattern_triples, response, re.DOTALL)
    new_predicates_match = re.search(pattern_new_predicates, response, re.DOTALL)
    
    df_cost_stat = cost_stat(completion,df_cost_stat)
    
    return (
        completion,
        hypothesis_match.group(1).strip() if hypothesis_match else None,
        keywords_match.group(1).strip() if keywords_match else None,
        triples_match.group(1).strip() if triples_match else None,
        new_predicates_match.group(1).strip() if new_predicates_match else None,
        df_cost_stat
    )

def extract_hypothesis_using_gpt4o(prompt,system_prompt=system_prompt_marketing,api_key=gpt4_api_key,model='GPT4O',temperature=0.3,max_completion_tokens=300):

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
    model=model, # model = "deployment_name"
    messages = message_text,
    temperature=temperature,
    max_tokens=max_completion_tokens,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
    )
    return process_gpt_response(completion)



def extract_hypothesis_using_gpt4_o1_mini(prompt,system_prompt=system_prompt_marketing,api_key=gpt_o1_mini_api_key,model='o1-mini',temperature=0.3,max_completion_tokens=100):
    #Note: The openai-python library support for Azure OpenAI is in preview.
    endpoint = "https://dalle-declare.openai.azure.com/openai/deployments/4o-mini/chat/completions?api-version=2024-08-01-preview"

    # Define the headers
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    # Define the payload
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": max_completion_tokens
    }

    # Send the POST request
    response = requests.post(endpoint, headers=headers, json=payload)

    # Check the response
    if response.status_code == 200:
        print(f'Successfully get response from gpt4o-mini')
    else:
        print(f"Error: {response.status_code}, {response.text}")
    return process_gpt_response(response)


def extract_hypothesis_using_gpt4o_mini(prompt, df_cost_stat, system_prompt=system_prompt_marketing, api_key=gpt4o_mini_api_key_, model='o1-mini', temperature=0.3, max_completion_tokens=300):
    try:
        client = AzureOpenAI(
            azure_endpoint="https://declaregpt4.openai.azure.com/", 
            api_key=api_key,
            api_version="2024-08-01-preview",
        )

        message_text = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        completion = client.chat.completions.create(
            model='GPT4O-mini',  # model = "deployment_name"
            messages=message_text,
            temperature=temperature,
            max_tokens=max_completion_tokens,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

        if not completion or not completion.choices:
            print("Warning: Received empty or invalid response from API")
            return None, None, None, None, None, df_cost_stat

        try:
            return process_gpt_response(completion,df_cost_stat)
        except (AttributeError, IndexError) as e:
            print(f"Error processing GPT response: {str(e)}")
            return None,None, None, None, None, df_cost_stat

    except openai.AuthenticationError:
        print("Error: Authentication failed. Please check your API key.")
        return None,None, None, None, None, df_cost_stat
    except openai.RateLimitError:
        print("Error: Rate limit exceeded. Please try again later.")
        return None, None, None, None, None, df_cost_stat
    except openai.APIConnectionError:
        print("Error: Failed to connect to the API. Please check your internet connection.")
        return None,None, None, None, None, df_cost_stat
    except openai.BadRequestError as e:
        print(f"Error: Bad request - {str(e)}")
        return None, None, None, None, None, df_cost_stat
    except ValueError as e:
        print(f"Error: Invalid parameter value - {str(e)}")
        return None, None, None, None, None, df_cost_stat
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
        return None, None, None, None, None, df_cost_stat
    

price = {
    'gpt4o-mini': {
        'prompt': 0.15/1000000.0,
        'completion': 0.6/1000000.0
    }
}

def cost_stat(completion,df_cost_stat,price=price):

    prompt_cost = price['gpt4o-mini']['prompt'] * completion.usage.prompt_tokens
    completion_cost = price['gpt4o-mini']['completion'] * completion.usage.completion_tokens
    # total_cost = {'prompt': {'token_count': completion.usage.prompt_tokens,'cost':  prompt_cost}, 
    #               'completion': {'token_count': completion.usage.completion_tokens, 'cost':completion_cost},
    #               'total_cost': prompt_cost+completion_cost}
            
    new_row = pd.DataFrame([{
        'id': completion.id,
        'prompt_tokens': completion.usage.prompt_tokens,
        'completion_tokens': completion.usage.completion_tokens,
        'prompt_cost': prompt_cost,
        'completion_cost': completion_cost,
        'total_cost': prompt_cost+completion_cost
    }])
    df_cost_stat = pd.concat([df_cost_stat,new_row],ignore_index=True)
    return df_cost_stat
# ===================== KIMI ========================
def extract_hypothesis_using_kimi(prompt,api_key,temperature=0.3):
    client = OpenAI(
        api_key=api_key, # 在这里将 MOONSHOT_API_KEY 替换为你从 Kimi 开放平台申请的 API Key
        base_url="https://api.moonshot.cn/v1",
    )
    

    # print(prompt)
    completion = client.chat.completions.create(
        model = "moonshot-v1-8k",
        messages = [
            {"role": "system", "content": "You are a social science expert searching for new research topics to explore. To accomplish this, you need to conduct a comprehensive literature review to understand the current research landscape, including the challenges being addressed, the hypotheses being tested, the methodologies employed, and the specific conditions under which key findings are obtained."},
            {"role": "user", "content": prompt}
        ],
        temperature = temperature,
    )
    
    # 通过 API 我们获得了 Kimi 大模型给予我们的回复消息（role=assistant）
    # print(completion.choices[0].message.content)
    hypothesis, keywords, triples, new_predicates = process_response_kimi(completion.choices[0].message.content)
    return completion

def process_response_kimi(response):
    if type(response) != str:
        response = response.choices[0].message.content  
        
    matches_hypothesis = re.findall(r'Hypothesis:\s*(.*?)(?=\#)', response, re.DOTALL)
    matches_keywords = re.findall(r'Keywords:\s*(.*?)(?=\#)', response, re.DOTALL)
    matches_triples = re.findall(r'Triples:\s*(.*?)(?=\#)', response, re.DOTALL)
    matches_new_predicates =  re.findall(r'New_predicates:\s*(.*?)(?=\#)', response, re.DOTALL)

    if len(matches_hypothesis) >0:
        hypothesis = matches_hypothesis[0]
    else:
        hypothesis = None
    if len(matches_keywords) >0:
        keywords = matches_keywords[0]
    else:
        keywords = None
    if len(matches_triples) >0:
        triples = matches_triples[0]
    else:
        triples = None
    if len(matches_new_predicates) >0:
        new_predicates = matches_new_predicates[0]
    else:
        new_predicates = None        
    return hypothesis, keywords, triples, new_predicates



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
