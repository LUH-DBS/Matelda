from multiprocessing.pool import ThreadPool
import requests
import wget 

def download(url):
    try:
        output_dir = './downloaded_files/'
        filename = wget.download(url, output_dir)
        print(filename)
    except Exception as e:
        print(e)

pool = ThreadPool(128)

response = requests.get('https://catalog.data.gov/api/3/action/package_search?rows=250000')
assert response.status_code == 200
results = response.json()['result']['results']
csv_related_links = []
for res in results:
    if res['url'] and '.csv' in res['url']:
        csv_related_links.append(res['url'])
    else:
        for resource in res['resources']:
            if resource['url'] and '.csv' in resource['url']:
                csv_related_links.append(resource['url'])

pool.map(download, csv_related_links)   