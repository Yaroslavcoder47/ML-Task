import requests as req
import csv
import os
from bs4 import BeautifulSoup
import re


directory = os.path.join(os.getcwd(), "data\\URL_list.csv")
output_directory = os.path.join(os.getcwd(), "data\\")

def get_data_from_file(name):
    with open(f'{output_directory}{name}', 'r', encoding='UTF-8') as reader:
        return reader.read()

def main():
    data_links = get_data_links()
    write_data_to_file(data_links)



def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script"]):
        script.extract()
    return soup.get_text()

def get_data_links():
    result = []
    with open(directory, newline="") as input_file:
        reader = csv.reader(input_file)
        for i, row in enumerate(reader):
            if i >= 20:
                break
            result.append(row)
    return result

def write_data_to_file(data_links):
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

# def clean_data():
#     cleaned_data = ""
#     with open(output_directory + "web1.txt", 'r', encoding='UTF-8') as reader:
#         data = reader.read()
#         cleaned_data = re.sub(r'\s+', ' ', data).strip()
#     return cleaned_data
