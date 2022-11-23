import json
import requests
import threading
import urllib 
import urllib.request
import urllib.parse as parse
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os 
import wget 


def download(url):
    output_dir = '/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/Sandbox_Generation/downloaded_files'
    filename = wget.download(url, output_dir)
    print(filename)


def scraper(url):
    html = urllib.request.urlopen(url).read() 
    soup = BeautifulSoup(html)
    csv_urls = []
    # Retrieve all of the anchor tags
    tags = soup('a')
    for tag in tags:
        href = (tag.get('href', None))
        if href and href.endswith(".csv"):
            csv_url = parse.urljoin(url, href)
            csv_urls.append(csv_url)
    return csv_urls

try:
    # scraper('https://bulkdata.uspto.gov/data/patent/historical/2014/')
    # Fetsching all packages with the CSV tag 
    response = requests.get('http://catalog.data.gov/api/3/action/tag_show?id=csv&include_datasets=True&rows=200000')
    assert response.status_code == 200
    packages = response.json()['result']['packages']
    package_ids = dict()
    for package in packages:
        package_ids[package['id']] = package['tags']

    # Fetsching links to the resources  
    resource_links = dict()

    for package_id in list(package_ids.keys()):
        response = requests.get('http://catalog.data.gov/api/3/action/package_show?id={}&rows=200000'.format(package_id))
        assert response.status_code == 200
        resources = response.json()['result']['resources']
        resource_links[package_id] = []
        for resource in resources:
            resource_links[package_id].append(resource['url'])

    accepted_ext = ['.csv', '.zip', '.tar.gz', '.tar']

    for package in list(resource_links.keys()):
        links = resource_links[package]
        for i,url in enumerate(links):
            if any(ext in url for ext in accepted_ext):
                threading.Thread(target = download, args = [url]).start()
            else:
                csv_urls = scraper(url)
                for j, iurl in enumerate(csv_urls):
                    threading.Thread(target = download, args = [iurl]).start()
            
except Exception as e:
    print(e)


