import requests as req
import csv
import os
from bs4 import BeautifulSoup
import re

directory = os.path.join(os.getcwd(), "data\\URL_list.csv")
output_directory = os.path.join(os.getcwd(), "data\\")

# variable flag is used to determine whether to extract extract data and write to files (false) or extract data by link and return them (true)

def main_prepare_data(web_link : str, flag : bool) -> str:
    if flag:
        return extract_info_by_link(web_link)
    data_links = get_data_links()
    return ""
    

def get_data_links() -> list:
    result = []
    with open(directory, newline="") as input_file:
        reader = csv.reader(input_file)
        for i, row in enumerate(reader):
            if i >= 20:
                break
            result.append(row)
    return result

def get_data_from_file(name : str) -> str:
    with open(f'{output_directory}{name}', 'r', encoding='UTF-8') as reader:
        return reader.read()


def extract_text_from_html(html_content) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script"]):
        script.extract()
    return soup.get_text()


def extract_info_by_link(html_url) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        request = req.get(html_url, headers=headers)
        data = extract_text_from_html(request.text)
        cleaned_data = re.sub(r'\s+', ' ', data).strip()
        return cleaned_data
    except req.exceptions.RequestException as e:
        return ""


def extract_info_to_file(data_links : list):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    for i, link in enumerate(data_links):
        try:
            request = req.get(link[0], headers=headers)
            with open(f'{output_directory}web{i}.txt', 'w', encoding='utf-8') as output_file:
                data = extract_text_from_html(request.text)
                clean_data = re.sub(r'\s+', ' ', data).strip()
                output_file.write(clean_data)
        except req.exceptions.RequestException as e:
            with open(f'{output_directory}web{i}.txt', 'w', encoding='utf-8') as output_file:
                output_file.write("")
            continue