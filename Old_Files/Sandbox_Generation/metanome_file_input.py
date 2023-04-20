import json
from matplotlib.pyplot import get
import requests
import time
import os
import sys

BASE_URL='http://localhost:8081/api'

def get_all_file_inputs():
    return requests.request("GET", f'{BASE_URL}/file-inputs').json()

def delete_all_file_inputs():
    for file_input in get_all_file_inputs():
        response = requests.request("DELETE", f'{BASE_URL}/file-inputs/delete/{file_input["id"]}')
        if response.status_code != 204:
            print('Can\'t delete the input file. file_id: file_input["id"]')
            sys.exit(1)
        print(f'file input deleted. id: {file_input["id"]}')
    
def load_input_file_to_metanome(input_file): 
    headers = {
        'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "id": 1,
        "type": "fileInput",
        "name": os.path.basename(input_file),
        "fileName": os.path.abspath(input_file),
        "separator": ",",
        "quoteChar": "'",
        "escapeChar": "\\",
        "skipLines": "0",
        "strictQuotes": False,
        "ignoreLeadingWhiteSpace": True,
        "hasHeader": True,
        "skipDifferingLines": False,
        "comment": "",
        "nullValue": ""
    })

    response = requests.request("POST", f'{BASE_URL}/file-inputs/store', headers=headers, data=payload)
    return response


def run_fd(file_name, file_id):
    headers = {
        'Content-Type': 'application/json'
    }
    payload = json.dumps(
        {
            "algorithmId":1,
            "executionIdentifier":"HyFD-1.2-SNAPSHOT.jar" + time.strftime("%Y-%m-%dT%H%M%S"),
            "requirements":[
                {
                    "type":"ConfigurationRequirementRelationalInput",
                    "identifier":"INPUT_GENERATOR",
                    "required": True,
                    "numberOfSettings":1,
                    "minNumberOfSettings":1,
                    "maxNumberOfSettings":1,
                    "settings":[
                        {
                            "fileName": file_name,
                            "advanced": False,
                            "separatorChar":",",
                            "quoteChar":"\"",
                            "escapeChar":"\\",
                            "strictQuotes": False,
                            "ignoreLeadingWhiteSpace": True,
                            "skipLines":0,
                            "header": True,
                            "skipDifferingLines": False,
                            "nullValue":"",
                            "type":"ConfigurationSettingFileInput",
                            "id": file_id
                         }
                    ]
                },
                {
                    "type":"ConfigurationRequirementInteger",
                    "identifier":"MAX_DETERMINANT_SIZE",
                    "required":False,
                    "numberOfSettings":1,
                    "minNumberOfSettings":1,
                    "maxNumberOfSettings":1,
                    "settings":[
                        {
                        "type":"ConfigurationSettingInteger",
                        "value":1
                        }
                    ],
                    "defaultValues":[
                        -1
                    ]
                },
                {
                    "type":"ConfigurationRequirementInteger",
                    "identifier":"INPUT_ROW_LIMIT",
                    "required":False,
                    "numberOfSettings":1,
                    "minNumberOfSettings":1,
                    "maxNumberOfSettings":1,
                    "settings":[
                        {
                        "type":"ConfigurationSettingInteger",
                        "value":-1
                        }
                    ],
                    "defaultValues":[
                        -1
                    ]
                },
                {
                    "type":"ConfigurationRequirementBoolean",
                    "identifier":"NULL_EQUALS_NULL",
                    "required":True,
                    "numberOfSettings":1,
                    "minNumberOfSettings":1,
                    "maxNumberOfSettings":1,
                    "settings":[
                        {
                        "type":"ConfigurationSettingBoolean",
                        "value":True
                        }
                    ],
                    "defaultValues":[
                        True
                    ]
                },
                {
                    "type":"ConfigurationRequirementBoolean",
                    "identifier":"VALIDATE_PARALLEL",
                    "required":True,
                    "numberOfSettings":1,
                    "minNumberOfSettings":1,
                    "maxNumberOfSettings":1,
                    "settings":[
                        {
                        "type":"ConfigurationSettingBoolean",
                        "value":True
                        }
                    ],
                    "defaultValues":[
                        True
                    ]
                },
                {
                    "type":"ConfigurationRequirementBoolean",
                    "identifier":"ENABLE_MEMORY_GUARDIAN",
                    "required":True,
                    "numberOfSettings":1,
                    "minNumberOfSettings":1,
                    "maxNumberOfSettings":1,
                    "settings":[
                        {
                        "type":"ConfigurationSettingBoolean",
                        "value":True
                        }
                    ],
                    "defaultValues":[
                        True
                    ]
                }
            ],
            "cacheResults":True,
            "writeResults":False,
            "countResults":False,
            "memory":""
            })

    try:
        response = requests.request("POST", f'{BASE_URL}/algorithm-execution', headers=headers, data=payload, timeout=60)
        return response
    except Exception as e:
        print(e)
        return

def load_execution_result(execution_id):
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({
        "id": f'{execution_id}',
        "notDetailed": True})

    return requests.request("POST", f'{BASE_URL}/result-store/load-execution/{execution_id}/true', headers=headers, data=payload)

def get_result_count():
    return requests.request("GET", f'{BASE_URL}/result-store/count/Functional Dependency')

def get_result(result_count):
    return requests.request("GET", f'{BASE_URL}/result-store/get-from-to/Functional Dependency/Determinant/true/0/{result_count}')

def get_fd(file_path):
    delete_all_file_inputs()

    load_input_file_response = load_input_file_to_metanome(file_path)
    if load_input_file_response.status_code != 204:
        print(f'Can\'t load file to metanome. file: {file_path}')
        return
    
    file_input_response = get_all_file_inputs()[0]
    execution_response = run_fd(file_input_response['fileName'], file_input_response['id'])
    if execution_response.status_code != 200:
        print(f'Can\'t execute fd. file: {file_path}')
        return
    execution_response = execution_response.json()
    
    load_execution_result_response = load_execution_result(execution_response["id"])
    if load_execution_result_response.status_code != 204:
        print(f'Can\'t load the execution result. execution_id: {execution_response["id"]}')
        return
    
    result_count_response = get_result_count()
    if result_count_response.status_code != 200:
        print(f'Can\'t get the result count. execution_id: {execution_response["id"]}')
        return

    result_response = get_result(result_count_response.text)
    if result_response.status_code != 200:
        print(f'Can\'t get the result. execution_id: {execution_response["id"]}')
        return
    
    return result_response.json()

def run_metanome(file_path):
    return get_fd(file_path)