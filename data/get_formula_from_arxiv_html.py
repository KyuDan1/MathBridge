import os
import requests
from bs4 import BeautifulSoup
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


# Given category of arXiv, get arxiv paper id and output list.
def get_arxiv_papers_html_link(category, max_papers=5,):
    #category = math
    
    base_url = "https://arxiv.org"
    # https://arxiv.org/list/math/recent?skip=0&show=2000
    category_url = f"{base_url}/list/{category}/recent?skip=0&show=2000"
    print(category_url)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        response = session.get(category_url)
        if response.status_code != 200:
            print(f"Failed to retrieve the webpage: {category_url} - Status code: {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        paper_links = soup.find_all("a", title="View HTML", limit=max_papers)

        return paper_links

    except Exception as e:
        print(f"An error occurred while fetching {category_url}: {e}")
        return None


# If we get id of paper, get mathematics formula from html file
def download_math_expressions_with_context(url, file_name='math_expressions.jsonl'):
    url = url
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            math_tags = soup.find_all('math', alttext=True)

            expressions_with_context = []
            for tag in math_tags:
                alttext = tag['alttext'].strip()
                previous_text = tag.previous_sibling.strip() if tag.previous_sibling and isinstance(tag.previous_sibling, str) else ''
                next_text = tag.next_sibling.strip() if tag.next_sibling and isinstance(tag.next_sibling, str) else ''
                expression_with_context = {
                    'Previous Text': previous_text,
                    'Math Expression': alttext,
                    'Next Text': next_text
                }
                expressions_with_context.append(expression_with_context)

            unique_expressions = [dict(t) for t in {tuple(d.items()) for d in expressions_with_context}]
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

            with open(file_name, 'w', encoding='utf-8') as file:
                for expression in unique_expressions:
                    file.write(json.dumps(expression, ensure_ascii=False) + '\n')

            print(f'Formulae saved: {file_name}')
        else:
            print(f'Failed to retrieve the page: {url}')
            return -1
    except Exception as e:
        print(f'An error occurred while fetching {url}: {e}')
        return -1


def main():
    process_category(category='math', dir_name='./downloads')


def process_category(category, dir_name):
    count = 0
    while True:
        html_urls = get_arxiv_papers_html_link(category=category, max_papers=5)
        if not html_urls:
            break

        for url in html_urls:
            id = url.rstrip('/')
            try:
                download_math_expressions_with_context(url, file_name=f"{dir_name}/{id}_tex.jsonl")
                count += 1
                print(f"Processed {count} papers for {category}")
            except Exception as e:
                print(f"An error occurred while processing {url}: {e}")
                continue


    print(f"Completed processing for category {category}")
    print(html_urls)

if __name__ == "__main__":
    main()


import glob

def merge_jsonl_files(input_directory, output_file):
    jsonl_files = glob.glob(f'{input_directory}/*.jsonl')

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in jsonl_files:
            with open(file_name, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

    print(f'Merged {len(jsonl_files)} files into {output_file}')

dir_name = 'jsonl_files_ma'
merge_jsonl_files(dir_name, 'merged_arxiv_formulas.jsonl')
