import aiohttp
import asyncio
import json
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

# DeepSeek API ??
API_URL = "https://api.deepseek.com/v1/chat/completions"
# API_KEY = "sk-f63ba7612eb24eb59daf489918880527"  # ????? API Key
API_KEY = "sk-3c8c039115354c2da2208586cd1b2b55"  # ????? API Key

MAX_CONCURRENT_REQUESTS = 50  # ?????????
TIMEOUT_SECONDS = 1800  # 30????

async def query_deepseek(
    session: aiohttp.ClientSession,
    prompt: str,
    index: int,
    pbar: tqdm,
    model: str = "deepseek-reasoner",
    stream: bool = False,
    temperature: float = 0.0,  # ???? 0.7(???)
) -> Tuple[int, Dict[str, Any]]:
    """??????? DeepSeek API,?????(????)"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": temperature,  # ??????
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    try:
        async with session.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS),
        ) as response:
            # ??HTTP???
            if response.status != 200:
                return index, {
                    "response": None,
                    "error": f"HTTP??: {response.status}",
                    "raw_response": await response.text()
                }

            full_response = []
            async for line in response.content:
                if line.strip():
                    if stream and line.startswith(b"data: "):
                        decoded_line = line.decode("utf-8").strip()
                        if decoded_line.startswith("data: "):
                            data = decoded_line[6:]
                            if data != "[DONE]":
                                full_response.append(data)
                    else:
                        full_response.append(line.decode("utf-8"))

            # ??????
            try:
                parsed_response = (
                    [json.loads(chunk) for chunk in full_response] if stream
                    else json.loads("".join(full_response)) if full_response
                    else None
                )
                result = {
                    "response": parsed_response,
                    "error": None,
                    "raw_response": full_response
                }
            except json.JSONDecodeError as e:
                result = {
                    "response": None,
                    "error": f"JSON????: {str(e)}",
                    "raw_response": full_response
                }

    except Exception as e:
        result = {
            "response": None,
            "error": f"????: {str(e)}",
            "raw_response": None
        }
    finally:
        pbar.update(1)
        return index, result

async def fetch_all_answers(prompts: List[str]) -> List[Dict[str, Any]]:
    """?????? prompts,????????????"""
    pbar = tqdm(total=len(prompts), desc="????", unit="??")
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            query_deepseek(session, prompt, idx, pbar)
            for idx, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
    pbar.close()
    # ?????,???????????
    sorted_results = sorted(results, key=lambda x: x[0])
    return [result for idx, result in sorted_results]

def get_answers(prompts: List[str]) -> List[str]:
    """
    ????:?? prompts ??,????? answers ??
    ?? Jupyter Notebook ??? Python ??
    ??:
        prompts = ["??", "???"]
        answers = get_answers(prompts)
    """
    try:
        # ??????????(?????Python??)
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # ????????,?????(???????)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # ??????????(?Jupyter Notebook)
        try:
            import nest_asyncio
            nest_asyncio.apply()  # ????????
        except ImportError:
            raise ImportError(
                "?Jupyter???????nest_asyncio,???: pip install nest_asyncio"
            )
    
    answers = loop.run_until_complete(fetch_all_answers(prompts))
    
    formatted_answers = []
    for answer in answers:
        if answer["error"]:
            formatted_answers.append(f"ERROR: {answer['error']}")
            continue
        
        try:
            if answer["response"] is None:
                formatted_answers.append("ERROR: ???")
                continue
                
            if isinstance(answer["response"], list):  # ????
                content = ""
                for chunk in answer["response"]:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        content += chunk["choices"][0].get("delta", {}).get("content", "")
                formatted_answers.append(content if content else "ERROR: ?????")
            else:  # ?????
                if "choices" in answer["response"] and len(answer["response"]["choices"]) > 0:
                    formatted_answers.append(
                        answer["response"]["choices"][0].get("message", {}).get("content", "ERROR: ?????")
                    )
                else:
                    formatted_answers.append("ERROR: ??????")
        except Exception as e:
            formatted_answers.append(f"ERROR: ?????? - {str(e)}")
    
    return formatted_answers



# ????(??????????)
if __name__ == "__main__":
    test_prompts = [
        "???????????",
        "?Python?????????",
        "?????????",
    ]
    print("????...")
    answers = get_answers(test_prompts)
    for q, a in zip(test_prompts, answers):
        print(f"Q: {q}\nA: {a}\n{'-'*50}")