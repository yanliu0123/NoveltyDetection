import os
import time
from urllib.parse import quote
import requests # crossref

from scholarly import scholarly  # google scholar
from semanticscholar import SemanticScholar  # semantic scholar
from sklearn.feature_extraction.text import TfidfVectorizer #cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
import arxiv # arxiv
import pandas as pd


def compute_cosine_similarity(title, title_query):
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([title, title_query])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

# ===================== Crossref Search ======================

def search_crossref(title, email="yan010@e.ntu.edu.sg"):
    """
    Search paper information using Crossref API
    
    Args:
        title: Paper title to search
        email: Your email for polite API usage
    
    Returns:
        Dictionary with paper info or None if not found
    """
    # Encode title for URL
    encoded_title = quote(title)
    
    # Crossref API URL with email (polite pool)
    url = f"https://api.crossref.org/works?query.title={encoded_title}&rows=5"
    headers = {
        'User-Agent': f'PythonScript/1.0 (mailto:{email})'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for bad status
        
        data = response.json()
        items = data['message']['items']
        
        if not items:
            print(f"No results found for: {title}")
            return None
            
        # Get best match
        best_match = items[0]  # First result is usually best match
        return best_match
        
    except requests.exceptions.RequestException as e:
        print(f"Error searching Crossref: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

def process_titles_batch(titles, delay=1):
    """
    Process a batch of titles with rate limiting
    
    Args:
        titles: List of paper titles
        delay: Delay between requests in seconds
    
    Returns:
        List of results
    """
    results = []
    
    for i, title in enumerate(titles):
        print(f"\nProcessing {i+1}/{len(titles)}")
        print(f"Title: {title}")
        
        # Search Crossref
        result = search_crossref(title)
        
        if result:
            print(f"Found: {result['title']}")
            print(f"paperId: {result['DOI']}")
            results.append(result)
        
        # Rate limiting
        time.sleep(delay)
    
    return results

def process_crossref_result(seed_title,best_match,threshold=0.7,flg_print=False):
    """
    Process Crossref API result and extract paper information
    
    Args:
        results: JSON response from Crossref API
    
    Returns:
        Dictionary with processed paper information
    """
    try:        
        # Handle publication date
        date_parts = best_match.get('created', {}).get('date-parts', [[]])[0]
        if flg_print:   
            print(f'raw date_parts: {date_parts}')
        if date_parts and len(date_parts) >= 1:
            # Convert all parts to strings and pad with zeros if needed
            date_parts = [str(part).zfill(2) for part in date_parts]
            if len(date_parts) == 1:  # Only year
                publicationDate = f"{date_parts[0]}-01-01"
            elif len(date_parts) == 2:  # Year and month
                publicationDate = f"{date_parts[0]}-{date_parts[1]}-01"
            else:  # Full date
                publicationDate = '-'.join(date_parts[:3])
        else:
            publicationDate = None
            
        # Extract other information
        title = best_match.get('title', [None])[0]
        cos_sim = compute_cosine_similarity(seed_title, title)
        if flg_print:
            print(f'title_get: {title}')
            print(f'cosine_similarity: {cos_sim:.2f}')
        if cos_sim < threshold:
            if flg_print:   
                print(f'No best match found')
            return None,None,cos_sim
        else:   
            result = {
                'title': best_match.get('title', [None])[0],
                'paperId': best_match.get('DOI'),
                'published_date': publicationDate,
                'type': best_match.get('type'),
                'citations': best_match.get('is-referenced-by-count', 0),
                'authors': [
                    f"{author.get('given', '')} {author.get('family', '')}".strip() 
                    for author in best_match.get('author', [])
                ],
                'abstract': best_match.get('abstract'),
                'venue': best_match.get('container-title')
            }
            
            if result['abstract']:
                result['abstract'] = result['abstract'].replace('<jats:p>','').replace('</jats:p>','').strip()
            # print(f"Processed date: {publicationDate}")
            return best_match,result,cos_sim
            
    except Exception as e:
        if flg_print:   
            print(f"Error processing result: {str(e)}")
        return None,None,None

def search_and_process_using_crossref(title,flg_print=False,threshold=0.7):
    try:
        # Search Crossref
        results = search_crossref(title)
        if not results:
            return None,None,None
            
        # Process results
        best_match,paper_info,cos_sim = process_crossref_result(title,results,threshold=threshold)
        if paper_info:
            if flg_print:
                print(f"Found paper: {paper_info['title']}")
                print(f"\tPublished: {paper_info['published_date']}")
                print(f"\tAuthors: {', '.join(paper_info['authors'])}")
                print(f"\tDOI: {paper_info['DOI']}")
                print(f"\tType: {paper_info['type']}")
                print(f"\tCitations: {paper_info['citations']}")
                print(f"\tAbstract: {paper_info['abstract']}")
        return best_match,paper_info,cos_sim
        
    except Exception as e:
        if flg_print:
            print(f"Error processing {title}: {str(e)}")
        return None,None,None

# ===================== Arxiv Search ======================

def find_best_arxiv_match(query_title, similarity_threshold=0.7, max_results=10,flg_print=False):
    """
    Find the best matching paper from arXiv using cosine similarity.
    
    Args:
        query_title: Title to search for
        similarity_threshold: Minimum similarity score to consider a match
        max_results: Maximum number of results to check
    
    Returns:
        tuple: (best_match, similarity_score) or (None, 0) if no good match found
    """
    try:
        # Initialize arXiv client
        client = arxiv.Client()
        search = arxiv.Search(
            query=query_title,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Get results
        results = list(client.results(search))
        if not results:
            return None, 0
            
        # Prepare titles for comparison
        titles = [r.title for r in results]
        if flg_print:
            for i,title in enumerate(titles):
                print(f'{i+1}/{len(titles)}: {title}')
        
        # Calculate cosine similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query_title] + titles)
        
        # Compare query with each result
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Find best match
        best_idx = similarities[0].argmax()
        best_score = similarities[0][best_idx]
        
        # Print debug info
        if flg_print:
            print(f"\nQuery: {query_title}")
            print(f"Best match: {titles[best_idx]}")
            print(f"Similarity: {best_score:.3f}")
        
        # Return best match if it meets threshold
        if best_score >= similarity_threshold:
            return results[best_idx], best_score
        else:
            return None, best_score
            
    except Exception as e:
        print(f"Error searching arXiv: {str(e)}")
        return None, 0
    
def process_arxiv_response(best_match,flg_print=False):
    paper_info = {}
    return paper_info

def search_and_process_using_arxiv(seed_title,flg_print=False,similarity_threshold=0.8):
    best_match,cosine_sim = find_best_arxiv_match(seed_title,similarity_threshold=similarity_threshold, max_results=10,flg_print=flg_print)
    paper_info = process_arxiv_response(best_match)
    return best_match,paper_info,cosine_sim

# ===================== Semantic Scholar Search ======================
def find_best_matching_paper_semantic_scholar(query_title, similarity_threshold=0.8, max_results=10,flg_print=False,semantic_API_Key='LzT0Zeckj52OyygQ0BuVdaFVpEenIyXS4s74WHrs',limit=5):
    """
    Search for papers using Semantic Scholar and find the best match using cosine similarity.
    
    Args:
        query_title (str): The title to search for
        similarity_threshold (float): Minimum similarity score to consider a match (0-1)
        max_results (int): Maximum number of results to retrieve from Semantic Scholar
    
    Returns:
        tuple: (best_matching_paper, similarity_score) or (None, 0) if no good match found
    """
    try:
        # Initialize Semantic Scholar client
        if flg_print:
            print(f'Seting up Semantic Scholar client...')

        sch = SemanticScholar(api_key=semantic_API_Key)  # Replace with your API key

        if flg_print:
            print('Done')
            print(f'Searching for: {query_title}...', end='')
        # Search for papers
        results = sch.search_paper(
            query=query_title,
            limit=max_results,
            fields=['title', 'abstract', 'year', 'citationCount', 'authors','references','tldr','publicationDate','openAccessPdf']
        )
        
        if flg_print:
            print(f'Found {len(results)} results')
        if not results:
            # print(f"No results found for: {query_title}")
            return None, 0
        
        # Prepare titles for comparison
        search_titles = []
        for i in range(len(results)):
            search_titles.append(results[i].title)
        # print('finished preparing titles')
        all_titles = [query_title] + search_titles
        
        # Calculate TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_titles)
        
        # Calculate similarity between query and each result
        query_vector = tfidf_matrix[0:1]  # Vector for query title
        result_vectors = tfidf_matrix[1:]  # Vectors for search results
        
        # Calculate cosine similarity for each result
        similarity_scores = cosine_similarity(query_vector, result_vectors)[0]
        
        # Create list of (paper, similarity) pairs
        paper_similarities = list(zip(results, similarity_scores))
        
        # Sort by similarity score in descending order
        paper_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get the best match that meets the threshold
        for paper, similarity in paper_similarities:
            if similarity >= similarity_threshold:
                if flg_print:   
                    print(f"\nFound match for: {query_title}")
                    print(f"Matched with: {paper.title}")
                    print(f"Similarity score: {similarity:.3f}")
                    print(f"Year: {paper.year}")
                    print(f"Citations: {paper.citationCount}")
                return paper, similarity
        
        # print(f"\nNo matches above threshold ({similarity_threshold}) for: {query_title}")
        # print(f"Best match ({paper_similarities[0][1]:.3f}): {paper_similarities[0][0].title}")
        return None, 0
    
    except Exception as e:
        print(f"Error processing title '{query_title}': {str(e)}")
        return None, 0

def process_semantic_scholar_response(best_match,cosine_sim):
    paperId = best_match.paperId if best_match.paperId else 'No result: semanticscholar'
    title = best_match.title if best_match.title else 'No result: semanticscholar'
    abstract = best_match.abstract if best_match.abstract else 'No result: semanticscholar'
    publicationDate = best_match.publicationDate if best_match.publicationDate else 'No result: semanticscholar'
    year = best_match.year if best_match.year else 'No result: semanticscholar'
    authors = best_match.authors if best_match.authors else 'No result: semanticscholar'
    venue = best_match.venue if best_match.venue else 'No result: semanticscholar'
    citationCount = best_match.citationCount if best_match.citationCount else 'No result: semanticscholar'
    publicationDate = publicationDate if publicationDate!='No result: semanticscholar' else year
    references = best_match.references if best_match.references else 'No result: semanticscholar'
    tldr = best_match.tldr if best_match.tldr else 'No result: semanticscholar'
    openAccessPdf = best_match.openAccessPdf if best_match.openAccessPdf else 'No result: semanticscholar'
    dic_to_update = {'paperId':paperId,
                    'title': title,
                    'abstract': abstract,
                    'publicationDate': publicationDate,
                    'references': references,
                    'tldr': tldr,
                    'authors': str(authors),
                    'venue': venue,
                    'citationCount': citationCount,
                    'cosine_sim': round(cosine_sim,2),
                    'openAccessPdf': openAccessPdf}
    return dic_to_update

def search_and_process_using_semantic_scholar(seed_title,flg_print=False,similarity_threshold=0.8):
    best_match,cosine_sim = find_best_matching_paper_semantic_scholar(seed_title,similarity_threshold=similarity_threshold, max_results=10,flg_print=False)
    if best_match:
        paper_info = process_semantic_scholar_response(best_match,cosine_sim)
        return best_match,paper_info,cosine_sim
    else:
        return None,None,cosine_sim


def search_semantic_scholar_batch_with_paper_ids(paper_ids, flg_print=False, api_key='LzT0Zeckj52OyygQ0BuVdaFVpEenIyXS4s74WHrs', max_retries=3, batch_size=50):
    """
    Retrieve multiple papers from Semantic Scholar with retry logic and batching
    
    Args:
        paper_ids: List of paper IDs or DOIs
        flg_print: Whether to print debug information
        api_key: Semantic Scholar API key
        max_retries: Maximum number of retry attempts
        batch_size: Number of papers to request in each batch
        
    Returns:
        List of retrieved papers
    """
    # Initialize the API client
    sch = SemanticScholar(api_key=api_key)
    all_papers = []
    
    # Split paper_ids into smaller batches
    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i:i + batch_size]
        if flg_print:
            print(f"Processing batch {i//batch_size + 1} of {(len(paper_ids) + batch_size - 1)//batch_size}")
        
        # Retry logic for each batch
        for attempt in range(max_retries):
            try:
                # Retrieve papers for current batch
                papers = sch.get_papers(batch)
                all_papers.extend(papers)
                
                # Print retrieved papers if flag is set
                if flg_print:
                    for paper in papers:
                        title = paper.title
                        authors = [author['name'] for author in paper.authors]
                        publicationDate = paper.publicationDate.strftime('%Y-%m-%d') if paper.publicationDate else paper.year
                        citationCount = paper.citationCount
                        abstract = paper.abstract
                        print(f"\t\t\t[Retrieved] Title: {title}")
                        print(f"\t\t\t[Retrieved] Authors: {authors}")
                        print(f"\t\t\t[Retrieved] Year: {publicationDate}")
                        print(f"\t\t\t[Retrieved] Citation Count: {citationCount}")
                        print(f"\t\t\t[Retrieved] Abstract: {abstract}")
                        print('\t\t\t','-' * 50)
                
                # If successful, break out of retry loop
                break
                
            except GatewayTimeoutException as e:
                if attempt < max_retries - 1:
                    if flg_print:
                        print(f"Gateway timeout on attempt {attempt + 1}. Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    if flg_print:
                        print(f"Failed to retrieve batch after {max_retries} attempts")
                    raise
            except Exception as e:
                if flg_print:
                    print(f"Error retrieving batch: {str(e)}")
                raise
    
    return all_papers

# field_count = {}
# for field in all_fields:
#     # Count occurrences where the field appears in s2FieldsOfStudy
#     field_count[field] = df_all['s2FieldsOfStudy'].str.contains(field).sum()
#     if field_count[field] > 0:
#         no += 1