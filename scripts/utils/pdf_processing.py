import re
from collections import Counter
import calendar

import pandas as pd
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFSyntaxError
from pdfminer.high_level import extract_pages
from pdfminer.high_level import extract_text

# pdfminer
def font_size_counter(pdf_path,top_k=2):
    height_list = []
    for i,page_layout in enumerate(extract_pages(pdf_path)):
        for j,element in enumerate(page_layout):
            height_list.append(round(element.height,2))

    counter = Counter(height_list)
    print(counter)
    top_2 = counter.most_common(top_k)
    print(top_2)

    return top_2

def read_pdf_metadata_pdfminer(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            parser = PDFParser(file)
            try:
                document = PDFDocument(parser)
                metadata = document.info[0] if document.info else {}
                full_text = extract_text(pdf_path)
                metadata['Abstract'] = full_text.split('Abstract\n')[-1].split('\n\n')[0].replace('\n','')
                return metadata
            except PDFSyntaxError:
                print(f"Invalid PDF format for {pdf_path}")
                return {}
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        return {}
    

def process_pdf_pdfminer(pdf_path,flg_print=False): # some mistakes when extract the full text
    target_info = ['Author','Title','Keywords','Abstract','References','Venue']
    with open(pdf_path, 'rb') as file:
        parser = PDFParser(file)
        document = PDFDocument(parser)
        # test = document.outline
        # print(test)
        metadata = document.info[0] if document.info else {}
        full_text = extract_text(pdf_path)
        # text_references = re.split(r'References\n',full_text,1)[-1]
        text_references = full_text.split('References\n')[-1]
        clean_references_text = text_references
        # clean_references_text = clean_text(text_references)

        metadata['Abstract'] = full_text.split('Abstract\n')[-1].split('\n\n')[0].replace('\n','')
        metadata['References'] = extract_references(clean_references_text,flg_print=flg_print)
        # # remove the references part from the full text
        full_text = full_text.split('References\n')

        full_text = '\n'.join(full_text[:-1]) if len(full_text) > 1 else full_text[0]
        clean_full_text = clean_text(full_text)
        
        metadata['inline_citations'] = extract_inline_citations(clean_full_text)
        metadata['matched_references'],df_error_log_inline_citation_not_found = match_citations_to_references(metadata['inline_citations']['sorted_inline_citations'],metadata['References'])

        for key in metadata.keys(): 
            if type(metadata[key]) == bytes:
                metadata[key] = metadata[key].decode('utf-8', errors='ignore')
            else:
                pass
            
        if flg_print:
            print('-'*20, 'Metadata', '-'*20)
            for key in metadata.keys():
                print(f'{key} = {metadata[key]}')
            print()
    return full_text, clean_references_text, metadata,df_error_log_inline_citation_not_found

# general, pd, counter, re
def clean_text(text):
    text = re.sub(r'\.\s*\n\n', '.****', text)
    text = text.replace('-\n', '')
    text = re.sub(r'\.\s*\n', '{{KEEP_NEWLINE}}', text)
    # Replace all other newlines with space
    text = re.sub(r'\n', ' ', text)
    # Restore kept newlines
    text = text.replace('{{KEEP_NEWLINE}}', '.\n')
    text = text.replace('.****', '.\n\n')
    text = text.replace('- ', '')
    # text = text.replace('  ', ' ')
    text = re.sub(r'([a-z])([A-Z][a-z]+)', r'\1 \2', text)

    # text = re.sub(r'https?://\S+[^]*\.', ',', text)     
    return text


def extract_references(text,flg_print=False,seperator='\n\n'):
    # df_references = pd.DataFrame(columns=['path_to_paper','authors','year','title','venue','type','raw_text'])
    
   
    df_references = extract_references_marketing(text,flg_print=flg_print,seperator=seperator)
    # print(df_references.head())
    none_titles = df_references['title'].isna().sum()
    total_rows = len(df_references)
    percentage = (none_titles / total_rows) * 100
    
    # if total_rows<10 or percentage<50:
    #     print(f'Failed Extraction: {percentage:.2f}% using marketing style')
    #     df_references = extract_references_nlp(text,seperator='\n\n')
    # none_titles = df_references['title'].isna().sum()
    # total_rows = len(df_references)
    if total_rows<10 or percentage>50:
        percentage = (none_titles / total_rows) * 100
        print(f'Failed Extraction: {percentage:.2f}% using marketing styles; ',end='')

        # retry using nlp style
        df_references = extract_references_nlp(text,flg_print=flg_print,seperator=seperator)
        # print(df_references.head())
        none_titles = df_references['title'].isna().sum()
        total_rows = len(df_references)
        percentage = (none_titles / total_rows) * 100
        if total_rows<10 or percentage>50:
            print(f'Failed Extraction: {percentage:.2f}% using nlp style')
        else:
            print(f'Success Extraction: {100-percentage:.2f}% using nlp style')
    else:
        print(f'Success Extraction: {100-percentage:.2f}% using marketing style')
    return df_references


def match_citations_to_references(citation_tuples, ref_df):
    # matched_df = pd.DataFrame(columns=['inline_citation','frequency','matched_references'+ref_df.columns.tolist()])
    matched_df = ref_df.copy()
    matched_df['inline_citation'] = None
    matched_df['frequency'] = None
    df_error_log_inline_citation_not_found = pd.DataFrame(columns=['path_to_paper','inline_citation','frequency'])
    
    for citation_raw, count in citation_tuples:
        # Extract authors from citation
        citation = re.sub(r'(\d{4}[a-d]{0,1}),.*', r'\1', citation_raw)

        citation = citation.replace('et al.','').replace(',','').replace(' and ','').replace('p. ','p.').replace('  ', ' ').strip()
        citation = re.sub(r'([a-z])([A-Z])', r'\1 \2', citation)
        
        # print(citation)

        search_strings = []
        flg_book = False
        for s in citation.split(' '):
            if 'p.' not in s:
                search_strings.append(s.lower())
            else:
                flg_book = True
                pass
        # print(search_strings)
        def contains_all(text):
            # text = text.lower()
            return all(s in text for s in search_strings)
        matching_indices = ref_df[ref_df['raw_text'].apply(contains_all)].index.tolist()
        if len(matching_indices) == 0:
            df_error_log_inline_citation_not_found = pd.concat([df_error_log_inline_citation_not_found,pd.DataFrame({'path_to_paper':[None],'inline_citation':[citation_raw],'frequency':[count]})],ignore_index=True)
            # print(matching_indices)
        matched_df.loc[matching_indices,'inline_citation'] = citation_raw
        matched_df.loc[matching_indices,'frequency'] = count
        # print(matching_indices)
    
    return matched_df,df_error_log_inline_citation_not_found


def extract_inline_citations_style1(text):
    total_inline_citations = []

    pattern_author_year =  r'[A-Z][a-z]+\s+and\s+[A-Z][a-zA-Z]+\s\(\d{4},\sp\.\s\d+\)'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)

    # 1 author year paper: Heide (1994, p. 81)
    pattern_author_year =  r'[A-Z][a-zA-Z]+\s\(\d{4},\sp\.\s\d+\)'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)

    # # 1 author year pape Lokker (1995, p. 59)
    # pattern_author_year = r'(?:\()?([A-Z][a-zA-Z]+)\s\((\d{4}),\sp\.\s(\d+)\)'
    # total_inline_citations += re.findall(pattern_author_year, text)
    # text = re.sub(pattern_author_year, '', text)

    # 2 author year page
    pattern_author_year = r'\(([A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s\d{4},\sp\.\s\d+)\)'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)

    # 1 author year page
    pattern_author_year = r'\(([A-Z][a-z]+\s\d{4},\sp\.\s\d+)\)'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)
    #. 4 author 
    pattern_author_year = r'[A-Z][a-z]+,\s+[A-Z][a-z]+,\s+[A-Z][a-z]+,\s+and\s+[A-Z][a-z]+\s+(?:\(\d{4}\)|\d{4})'
    total_inline_citations += re.findall(pattern_author_year, text)

    text = re.sub(pattern_author_year, '', text)
    #. 3 author 
    pattern_author_year = r'[A-Z][a-z]+,\s+[A-Z][a-z]+,\s+and\s+[A-Z][a-z]+\s+(?:\(\d{4}\)|\d{4})'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)
    #. 2 author 
    pattern_author_year = r'[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s+(?:\(\d{4}\)|\d{4})'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)
    #. 1 author 
    pattern_author_year = r'[A-Z][a-z]+\s+(?:\(\d{4}\)|\d{4})'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)

    pattern_author_year = r'[A-Z][a-z]+\s+et\s+al\.\s+(?:\(\d{4}\)|\d{4})'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)
    # print(text)
     # 2 author, narrative citation with specific element, A and B (2020, p. 123)
    pattern_author_year = r'[A-Z][a-z]+\s+and\s+[A-Z][a-zA-Z]+\s\(\d{4},\s[^)]+\)'  
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)

    # 1 author, narrative citation with specific element, B (2020, p. 123)
    pattern_author_year = r'[A-Z][a-zA-Z]+\s\(\d{4},\s[^)]+\)'  
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)
    return total_inline_citations


def extract_inline_citations_style2(text):
    total_inline_citations = []

    pattern_author_year =  r'[A-Z][a-z]+\s+and\s+[A-Z][a-zA-Z]+\s\(\d{4}[a-d]{0,1},\sp\.\s\d+\)'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)

    # 1 author year paper: Heide (1994, p. 81)
    pattern_author_year =  r'[A-Z][a-zA-Z]+\s\(\d{4}[a-d]{0,1},\sp\.\s\d+\)'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)

    # # 1 author year pape Lokker (1995, p. 59)
    # pattern_author_year = r'(?:\()?([A-Z][a-zA-Z]+)\s\((\d{4}),\sp\.\s(\d+)\)'
    # total_inline_citations += re.findall(pattern_author_year, text)
    # text = re.sub(pattern_author_year, '', text)

    # 2 author year page
    pattern_author_year = r'\(([A-Z][a-z]+\s+and\s+[A-Z][a-z]+,?\s*\d{4}[a-d]{0,1},\sp\.\s\d+)\)'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)

    # 1 author year page
    pattern_author_year = r'\(([A-Z][a-z]+,?\s*\d{4}[a-d]{0,1},\sp\.\s\d+)\)'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)
    #. 4 author 
    pattern_author_year = r'[A-Z][a-z]+,\s+[A-Z][a-z]+,\s+[A-Z][a-z]+,\s+and\s+[A-Z][a-z]+,?\s*(?:\(\d{4}[a-d]{0,1}\)|\d{4}[a-d]{0,1})'
    total_inline_citations += re.findall(pattern_author_year, text)

    text = re.sub(pattern_author_year, '', text)
    #. 3 author 
    pattern_author_year = r'[A-Z][a-z]+,\s+[A-Z][a-z]+,\s+and\s+[A-Z][a-z]+,?\s*(?:\(\d{4}[a-d]{0,1}\)|\d{4}[a-d]{0,1})'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)
    #. 2 author 
    pattern_author_year = r'[A-Z][a-z]+\s+and\s+[A-Z][a-z]+,?\s*(?:\(\d{4}[a-d]{0,1}\)|\d{4}[a-d]{0,1})'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)
    #. 1 author 
    pattern_author_year = r'[A-Z][a-z]+,?\s*(?:\(\d{4}[a-d]{0,1}\)|\d{4}[a-d]{0,1})'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)

    pattern_author_year = r'[A-Z][a-z]+\s+et\s+al\.,?\s*(?:\(\d{4}[a-d]{0,1}\)|\d{4}[a-d]{0,1})'
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)
    # print(text)
     # 2 author, narrative citation with specific element, A and B (2020, p. 123)
    pattern_author_year = r'[A-Z][a-z]+\s+and\s+[A-Z][a-zA-Z]+,?\s*\(\d{4}[a-d]{0,1},\s[^)]+\)'  
    total_inline_citations += re.findall(pattern_author_year, text)
    text = re.sub(pattern_author_year, '', text)

    # 1 author, narrative citation with specific element, B (2020, p. 123)
    pattern_author_year = r'[A-Z][a-zA-Z]+,?\s*\(\d{4}[a-d]{0,1},\s[^)]+\)'  
    total_inline_citations += re.findall(pattern_author_year, text)

    text = re.sub(pattern_author_year, '', text)

    total_inline_citations = [citation.replace('\n',' ').strip() for citation in total_inline_citations]
    return total_inline_citations
    
    # # total_inline_citations = extract_inline_citations_style2(text)
    # counter = Counter(total_inline_citations)
    # sorted_total_inline_citations = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # print('-'*20, 'extract_inline_citations_style2', '-'*20)
    # print(f'{len(total_inline_citations)} inline citations found; {len(sorted_total_inline_citations)} unique inline citations')
    # for citation, count in sorted_total_inline_citations:
    #     print(f'{citation} = {count}')
    # print()
    # return {
    #     'raw_inline_citations': total_inline_citations,
    #     'sorted_inline_citations':sorted_total_inline_citations
    # }


def extract_inline_citations(text,flag_print=False):
    # pattern_author_year = r'(?:[A-Z][A-Za-z\s,]+(?:et al\.?)?\s+(?:\(\d{4}[a-z]?\)|\d{4}[a-z]?)(?:,\s*[A-Z][A-Za-z\s]+(?:et al\.?)?\s+(?:\(\d{4}[a-z]?\)|\d{4}[a-z]?))*)'# very very good
    # break into steps:
    total_inline_citations = []
    # 2 author year paper: Choi and Triantis (2010, p. 853)
    
    total_inline_citations = extract_inline_citations_style2(text)
    total_inline_citations = [citation.replace('\n',' ').replace('(', '').replace(')', '').strip() for citation in total_inline_citations]

    counter = Counter(total_inline_citations)
    sorted_total_inline_citations = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    if flag_print:
        print('-'*20, 'extract_inline_citations', '-'*20)
        print(f'{len(total_inline_citations)} inline citations found; {len(sorted_total_inline_citations)} unique inline citations')
        for citation, count in sorted_total_inline_citations:
            print(f'{citation} = {count}')
        print()

    return {
        'raw_inline_citations': total_inline_citations,
        'sorted_inline_citations':sorted_total_inline_citations
    }


def extract_references_marketing(text,flg_print=False,seperator='\n\n'):
    references = text.split(seperator)
    df_references = pd.DataFrame(columns=['path_to_paper','authors','year','title','venue','type','raw_text'])
    # print('1.\n', df_references,'\n')
    author_year_pattern = r'([A-ZÀ-ÿ][a-zA-ZÀ-ÿ,\s\-]+,\s?\d{4}\,)'
    author_years = re.findall(author_year_pattern,text)

    for author_year in author_years:
        text = text.replace(author_year,'##'+author_year)
    
    references = text.split('##')
    references = [x.replace('\n','').strip() for x in references]
    references = [x for x in references if len(x)>10]
    
    for i,x in enumerate(references):
        # x = x.decode('utf-8', errors='ignore')
        x = x.replace('\n','').strip()
        if flg_print:   
            print('-'*20,f'{i+1}/{len(references)}', '-'*20)
            print(x)
            print(len(x))
        # authors = x.split('\n')[0]
        year_pattern = r'(\(\d{4}\))'
        if re.search(year_pattern, x):
            parts = re.split(year_pattern, x, maxsplit=1)
            if len(parts) == 3:
                authors = parts[0].strip()
                year = parts[1]  # This is the '(YYYY)' part
                _remaining = parts[2].strip()
                # _remaining = parts[2].strip().replace(',”','.')
            if "”" in _remaining:
                doctype = 'paper'
                title,_remaining = re.split(r'”',_remaining,1)
            elif '.' in _remaining:
                doctype = 'book'
                title,_remaining = re.split(r'\.',_remaining,1)
            else:
                doctype = 'other'
                title = _remaining
            title = title.replace(',','').strip()
            title = title.replace('“','').replace('”','').strip()
            if ',' in _remaining:
                journal,_remaining = re.split(r',',_remaining,1)
            else:
                journal = _remaining
        elif len(x) <50:
                continue
        else:
            authors = None
            year = None
            title = None
            journal = None
            doctype = None
                 #cleaning
        path_to_paper = None
        df_references = pd.concat([df_references, pd.DataFrame({'path_to_paper': [path_to_paper], 'authors': [authors], 'year': [year], 'title': [title], 'venue': [journal],'type':[doctype],'raw_text':[x]})],ignore_index=True)
        if flg_print:
            print(f'authors = {authors}')
            print(f'year = {year}')
            print(f'title = {title}')
            print(f'journal = {journal}')
            print(f'_remaining = {_remaining}')
            print()
            # Drop rows where all columns are None/NaN
    df_references = df_references.dropna(how='all')
    # print('2.\n', df_references,'\n')

    # If you want to be more explicit about None values:
    # df_references = df_references[~df_references.isnull().all(axis=1)]
    df_references = df_references[df_references['raw_text'].apply(lambda x: x is not None)]
    # print('3.\n', df_references,'\n')

    return df_references


def is_date_citation(citation):
    months = []
    for i in range(1, 13):
        months.append(calendar.month_name[i])    # Full names
        months.append(calendar.month_abbr[i])    # Abbreviated names

    # Remove empty string from the list
    months = [m for m in months if m]

    # Create pattern
    month_year_pattern = r'^(?:' + '|'.join(months) + r')\s+\d{4}$'
    return bool(re.match(month_year_pattern, citation.strip(), re.IGNORECASE))


def parse_reference_style1(reference_text,flag_print=False):
    year_pattern = r'(\(\d{4}\))'
    if re.search(year_pattern, reference_text):
        parts = re.split(year_pattern, reference_text, maxsplit=1)
        if len(parts) == 3:
            authors = parts[0].strip()
            year = parts[1]  # This is the '(YYYY)' part
            _remaining = parts[2].strip()
            # _remaining = parts[2].strip().replace(',”','.')
            if "”" in _remaining:
                doctype = 'paper'
                title,_remaining = re.split(r'”',_remaining,1)
            elif '.' in _remaining:
                doctype = 'book'
                title,_remaining = re.split(r'\.',_remaining,1)
            else:
                doctype = 'other'
                title = _remaining
            title = title.replace(',','').strip()
            title = title.replace('“','').replace('”','').strip()
            if ',' in _remaining:
                journal,_remaining = re.split(r',',_remaining,1)
            else:
                journal = _remaining
            if flag_print:
                print(f'authors = {authors}')
                print(f'year = {year}')
                print(f'title = {title}')
                print(f'venue = {journal}')
                print(f'doctype = {doctype}')
            return {
                'authors': authors,
                'year': year,
                'title': title,
                'venue': journal,
                'doctype': doctype
            }
        elif len(reference_text) <50:
            return None
        else:
            return None
    return None


def parse_reference_style_acl(reference_text,flag_print=False):
    """
    Parse a reference into authors, year, title, and venue
    """
    # # Pattern to match the components
    # pattern = r"""
    #     ^(.+?)\.\s*                    # Authors (non-greedy, up to first period)
    #     (\d{4})\.\s*                   # Year
    #     ([^\.]+?)\.\s*                 # Title (non-greedy, up to period)
    #     (.*?(?:pages|pp\.)\s+\d+(?:[-–]\d+)?)?[,\.]?\s*  # Venue and pages
    #     ([^,\.]+(?:Press|Association|Publisher|Publishers))?\s*\.?$  # Publisher
    # """
    year_pattern = r'\.\s*(\d{4})\.'
    year = re.findall(year_pattern, reference_text)
    # match = re.match(year_pattern, reference_text, re.VERBOSE)
    if len(year) == 1:
        if int(year[0])>1800 and int(year[0])<2025:
            year = year[0]
        else:
            year = None
    elif len(year) == 0:
        year = None
    else:
        print('error, multiple year')
        year = None 
    # print(f'year = {year}')
    if year:
        authors, _remaining = reference_text.split(year+'.',1)
        # print(f'authors = {authors}')
        # print(f'_remaining = {_remaining}')
        last_question = _remaining.rfind('?')
        last_exclamation = _remaining.rfind('!')

        seperator = '?' if last_question > last_exclamation else '!'
        if last_question == -1 and last_exclamation == -1:
            seperator = '.'
            parts = _remaining.split('.',1)
            if len(parts) == 3:
                title = parts[0]
                venue = parts[1]
                publisher = parts[2]
            else:
                title = parts[0]
                venue = parts[1]
                publisher = None
        else:
            parts = _remaining.split(seperator,1)
            if len(parts) == 2:
                title = parts[0]
                _remaining = parts[1]
                parts = _remaining.split('.',1)
                if len(parts) == 2:
                    venue = parts[0]
                    publisher = parts[1]
                else:
                    venue = parts[0]
                    publisher = None
            else:
                title = None
                venue = None
                publisher = None
        # title, _remaining = re.split(f'(?<={seperator})', _remaining)         

        
        # Clean up the components
        authors = re.sub(r'([a-z])([A-Z])', r'\1 \2', authors).strip() if authors else None
        year = year.strip() if year else None
        title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title).strip() if title else None    
        venue = re.sub(r'([a-z])([A-Z])', r'\1 \2', venue).strip() if venue else None
        publisher = re.sub(r'([a-z])([A-Z])', r'\1 \2', publisher).strip() if publisher else None

        if flag_print:
            print(f'authors = {authors}')
            print(f'year = {year}')
            print(f'title = {title}')
            print(f'venue = {venue}')
            print(f'publisher = {publisher}')
        return {
            'authors': authors,
            'year': year,
            'title': title,
            'venue': venue,
            'publisher': publisher
        }
    else:
        # authors = None
        # year = None
        # title = None
        # venue = None
        # publisher = None
        return None


def extract_references_nlp(text,flg_print=False,seperator='\n\n'):
    all_matches = []
    author_year_pattern = r'([A-ZÀ-ÿ][a-zA-ZÀ-ÿ,\s\-]+\.\s?\d{4}\.)'
    author_years = re.findall(author_year_pattern,text)

    for author_year in author_years:
        text = text.replace(author_year,'##'+author_year)
    
    references = text.split('##')
    references = [x.replace('\n','').strip() for x in references]
    references = [x for x in references if len(x)>10]


    count = 0
    df_references = pd.DataFrame(columns=['path_to_paper','authors','year','title','venue','type','raw_text'])
    for reference in references:
        reference = reference.strip()
        # print('-'*20,f'{count}/{len(references)}', '-'*20)
        # print(type(reference), len(reference))
        # print(f'reference = {reference}')

        parsed_reference = parse_reference_style_acl(reference)
        if parsed_reference:
            count+=1
            authors = parsed_reference['authors'] if parsed_reference['authors'] else ''
            year = parsed_reference['year'] if parsed_reference['year'] else ''
            title = parsed_reference['title'] if parsed_reference['title'] else ''
            venue = parsed_reference['venue'] if parsed_reference['venue'] else ''
            publisher = parsed_reference['publisher'] if parsed_reference['publisher'] else ''
            if flg_print:
                print('-'*20,f'{count}/{len(references)}', '-'*20)
                print(f'authors: {authors}')
                print(f'year: {year}')
                print(f'title: {title}')
                print(f'venue: {venue}')
                print(f'publisher: {publisher}')
                print()
            df_references = pd.concat([df_references, pd.DataFrame({'path_to_paper':None,'authors': [authors], 'year': [year], 'title': [title], 'venue': [venue],'type':['paper'],  'raw_text':[reference]})],ignore_index=True)
            # df_references = pd.concat([df_references, pd.DataFrame({'path_to_paper':None,'authors': [authors], 'year': [year], 'title': [title], 'venue': [journal],'type':[doctype],'raw_text':        [x]})],ignore_index=True)
        
    # all_matches.extend(author_years)

    
    # print(author_years) 
    df_references = df_references[df_references['raw_text'].apply(lambda x: x is not None)]

    return df_references
