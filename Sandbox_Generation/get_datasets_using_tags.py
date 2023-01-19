import json
from time import sleep
import requests
import threading
import urllib 
import urllib.request
import urllib.parse as parse
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os 
import wget 
import json
from multiprocessing.dummy import Pool as ThreadPool


accepted_ext = ['.CSV', '.csv', 'xls', 'XLSX', 'XLS', 'xlsx', '.zip', '.tar.gz', '.tar']
headers = requests.utils.default_headers()

headers.update(
    {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:101.0) Gecko/20100101 Firefox/101.0',
    }
)
pool = ThreadPool(128)

def download(url):
    try:
        output_dir = './downloaded_files/'
        filename = wget.download(url, output_dir)
        print(filename)
    except Exception as e:
        print(e)

def scraper(url):
    try:
        req = urllib.request.Request(url, headers = headers)
        html = urllib.request.urlopen(req).read() 
        soup = BeautifulSoup(html, features="xml")
        file_urls = []
        # Retrieve all of the anchor tags
        tags = soup('a')
        for tag in tags:
            href = (tag.get('href', None))
            if href and (href.endswith(tuple(accepted_ext))):
                file_url = parse.urljoin(url, href)
                file_urls.append(file_url)
        print("Retrieved file urls, {}".format(url))
        return file_urls
    except Exception as e:
        print(e)
try:

    # Fetsching all packages with the CSV tag 
    response = requests.get('http://catalog.data.gov/api/3/action/tag_show?id=csv&include_datasets=True&rows=200000', headers=headers)
    assert response.status_code == 200
    packages = response.json()['result']['packages']
    package_ids = dict()
    for package in packages:
        package_ids[package['id']] = package['tags']

    with open("packages_info.json", "w") as outfile:
            json.dump(package_ids, outfile)

    # Fetsching links to the resources  
    resource_links = dict()

    for package_id in list(package_ids.keys()):
        response = requests.get('http://catalog.data.gov/api/3/action/package_show?id={}&rows=200000'.format(package_id), headers=headers)
        assert response.status_code == 200
        resources = response.json()['result']['resources']
        resource_links[package_id] = []
        for resource in resources:
            resource_links[package_id].append(resource['url'])

    all_links = []
    for package in list(resource_links.keys()):
        for link in resource_links[package]:
            all_links.append(link)
    
    links_for_download = []

    datasets = []
    pages = []
    for i,url in enumerate(all_links):
        if any(ext in url for ext in accepted_ext):
            datasets.append(url)
        else:
            pages.append(url)

    print("{} datasets and {} pages".format(len(datasets), len(pages))) 

    links_for_download.extend(datasets)

    pages_links = pool.map(scraper, pages)
    links_for_download.extend(pages_links)

    pool.map(download, links_for_download)     

except Exception as e:
    print(e)


