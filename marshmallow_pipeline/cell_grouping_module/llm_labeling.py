import json
import logging
import os
import threading
from time import sleep
import time
import openai


def get_foundation_model_prediction(tables_tuples_dict, key, serialization_type = "json"):
    table_id = key[0]
    header = tables_tuples_dict[table_id]["header"]
    tuple = tables_tuples_dict[table_id]["tuples"][key[2]]
    ground_truth = tables_tuples_dict[table_id]["clean"][key[2]]
    attribute_idx = key[1]
    value = tuple[attribute_idx]
    value_gt = ground_truth[attribute_idx]
    if serialization_type == "tagbased":
        prompt = serialize_tuple_tag_based(key, header, tuple, -1)
    elif serialization_type == "json":
        prompt = serialize_tuple_json(key, header, tuple, -1)
    label = ask_model(prompt)
    logging.info("Asking Foundation Model: prompt: " + prompt + " Model label: " + str(label) + " Dirty Value: " + value + " Clean Value: " + value_gt)
    sleep(1)
    return label


def serialize_tuple_tag_based(tuple_key, header, tuple, user_label):
    attribute_idx = tuple_key[1]
    attribute_name = header[attribute_idx]
    tuple_serialized = "[SOT] "
    for col_idx, col in enumerate(header):
        tuple_serialized += "[COL] " + col + " [VAL] " + tuple[col_idx] 
    tuple_serialized += " [SA] " + attribute_name
    if user_label != -1:
        tuple_serialized += " [UL] " + str(user_label)
    tuple_serialized += " [EOT]\n" 
    return tuple_serialized

def serialize_tuple_json(tuple_key, header, tuple, user_label):
    attribute_idx = tuple_key[1]
    attribute_name = header[attribute_idx]
    tuple_dict = {"Tuple": {}, "Specified Attribute for Evaluation": attribute_name}
    for col_idx, col in enumerate(header):
        tuple_dict["Tuple"][col] = tuple[col_idx]
    if user_label != -1:
        tuple_dict["User Label for the specified attribute"] = user_label
    tuple_serialized = json.dumps(tuple_dict, indent=4)
    return tuple_serialized



def few_shot_prediction(tables_tuples_dict, keys, user_samples_dict, serialization_type = "json"):
    try:
        logging.info("FEW SHOT PREDICTION")
        few_shot_strings = "### Example Tuples:\n"
        for sample in user_samples_dict.keys():
            table_id = sample[0]
            header = tables_tuples_dict[table_id]["header"]
            tuple = tables_tuples_dict[table_id]["tuples"][sample[2]]
            user_label = user_samples_dict[sample]
            if serialization_type == "tagbased":
                tuple_serialized = serialize_tuple_tag_based(sample, header, tuple, user_label)
            elif serialization_type == "json":
                tuple_serialized = serialize_tuple_json(sample, header, tuple, user_label)
            few_shot_strings += tuple_serialized
        test_strings = "### Test Tuples:\n"
        test_values_dict = {"value": [], "value_gt": []}
        for key in keys:
            table_id = key[0]
            header = tables_tuples_dict[table_id]["header"]
            tuple = tables_tuples_dict[table_id]["tuples"][key[2]]
            ground_truth = tables_tuples_dict[table_id]["clean"][key[2]]
            attribute_idx = key[1]
            test_values_dict["value"].append(tuple[attribute_idx])
            test_values_dict["value_gt"].append(ground_truth[attribute_idx])
            if serialization_type == "tagbased":
                tuple_serialized = serialize_tuple_tag_based(key, header, tuple, -1)
            elif serialization_type == "json":
                tuple_serialized = serialize_tuple_json(key, header, tuple, -1)
            test_strings += tuple_serialized
        prompt = few_shot_strings + test_strings
        test_labels = ask_model_few_shot_comp(prompt, len(keys))
        logging.info("Finished few shot prediction")
        logging.info("Few shot prompt: " + prompt)
        results = ""
        for i in range(len(test_labels)):
            results += "Foundation Model label: " + str(test_labels[i]) + " Dirty Value: " + test_values_dict["value"][i] + " Clean Value: " + test_values_dict["value_gt"][i] + "\n"
        logging.info(results)
    except Exception as e:
        logging.error("Error in few_shot_prediction: " + str(e))
    return test_labels
   

def get_instraction_task_details(serialization_type, n_shots):
    if serialization_type == "tagbased":
        instruction = "### Task: Evaluate the semantic and syntactic alignment of specified attributes in test tuples. The tuples follow a structured format, where:\
                        [SOT] marks the start of a tuple.\
                        [COL] denotes an attribute name. \
                        [VAL] denotes the value of the attribute.\
                        [EOT] marks the end of a tuple and acts as a separator.\
                        [SA] indicates the attribute to be evaluated.\n"

        task_details = "### Task Details: Use the AI's internal knowledge to assess if the values for each [SA] \
                in the test tuples are correctly aligned, both semantically and syntactically, with their respective [COL].\
                For lesser-known or obscure details, focus on logical consistency and syntactic correctness."

        if n_shots > 0:
            instruction += "[UL] represents the user label for the value of the specified attribute, with 1 indicating an error and 0 indicating a correct value.\n"
            task_details += "Use the given examples as a reference for the expected format and content alignment."

        task_details += "###output Format: Provide \"only\" a 1-character output for each test tuple separated by comma, example output: 1, 0, with 1 indicating an error and 0 indicating a correct value.\n"
    
    elif serialization_type == "json":
        instruction = "### Task: Evaluate the semantic and syntactic alignment of specified attributes in test tuples.\n"
        
        task_details = "### Task Details: Use the AI's internal knowledge to assess if the values for each specified value \
                        in the test tuples are correctly aligned, both semantically and syntactically, with their respective attribute.\
                        For lesser-known or obscure details, focus on logical consistency and syntactic correctness."
        
        if n_shots > 0:
            task_details += "Use the given examples as a reference for the expected format and content alignment."

        task_details += "###output Format: Provide \"only\" a 1-character output for each test tuple separated by comma, example output: 1, 0, with 1 indicating an error and 0 indicating a correct value.\n"

    return instruction, task_details



def api_call_comp(prompt, response_container, n_shots, serialization_type = 'json'):
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        instruction, task_details = get_instraction_task_details(serialization_type, n_shots)
        full_prompt = f"{instruction}\n\n{prompt}\n\n{task_details}"
        
        response_container['completion'] = openai.Completion.create(
            # model="text-curie-001",  # Use a non-chat model
            model = "text-curie-001",
            prompt=full_prompt
            # max_tokens=100  # Adjust as needed
        )
    except Exception as e:
        response_container['exception'] = e

def ask_model_few_shot_comp(prompt, n_shots):
    max_retries = 5
    backoff_factor = 1
    retry_delay = 1
    timeout_seconds = 120

    for i in range(max_retries):
        response_container = {}
        thread = threading.Thread(target=api_call_comp, args=(prompt, response_container, n_shots, 'json'))
        thread.start()
        logging.info("Started thread for API call")
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            logging.error(f"API call didn't finish within {timeout_seconds} seconds, retrying...")
            thread.join()  # Ensure thread has completed before continuing or retrying
        else:
            if 'exception' in response_container:
                logging.error(f"Exception raised in API call: {response_container['exception']}")
            else:
                completion = response_container.get('completion')
                if completion:
                    # Adjust how the response is parsed for non-chat completions
                    responses = completion.choices[0].text.strip().split(",")
                    test_labels = []
                    logging.info("len responses: " + str(len(responses)))
                    logging.info("n_shots: " + str(n_shots))
                    if len(responses) >= n_shots:
                        for i in range(n_shots):
                            if responses[i].strip() == "1":
                                test_labels.append(1)
                                logging.debug(responses[i].strip())
                            else:
                                test_labels.append(0)
                                logging.debug(responses[i].strip())
                        logging.info("Got Answer from the Model")
                        return test_labels

        time.sleep(retry_delay)
        retry_delay *= backoff_factor

    logging.error("Error in ask_model_few_shot after maximum retries")
    return [0 for i in range(n_shots)]

def ask_model(prompt, serialization_type = 'json'):
    try:
        response_container = {}
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if serialization_type == "tagbased":
            instruction = "### Task: Evaluate the semantic and syntactic alignment of specified attributes in test tuples. The tuples follow a structured format, where:\
                    [SOT] marks the start of a tuple.\
                    [COL] denotes an attribute name. \
                    [VAL] denotes the value of the attribute.\
                    [EOT] marks the end of a tuple and acts as a separator.\
                    [SA] indicates the attribute to be evaluated.\n"
            
            task_details = "### Task Details: Use the AI's internal knowledge to assess if the values for each [SA] \
                            in the test tuples are correctly aligned, both semantically and syntactically, with their respective [COL].\
                        For lesser-known or obscure details, focus on logical consistency and syntactic correctness. \
                            ###output Format: Provide \"only\" a 1-character output for each test tuple separated by comma, example output: 1, 0, with 1 indicating an error and 0 indicating a correct value.\n"
        elif serialization_type == "json":
            instruction = "### Task: Evaluate the semantic and syntactic alignment of specified attributes in test tuples.\n"
            
            task_details = "### Task Details: Use the AI's internal knowledge to assess if the values for each specified value \
                            in the test tuples are correctly aligned, both semantically and syntactically, with their respective attribute.\
                        For lesser-known or obscure details, focus on logical consistency and syntactic correctness. \
                            ###output Format: Provide \"only\" a 1-character output for each test tuple separated by comma, example output: 1, 0, with 1 indicating an error and 0 indicating a correct value.\n"


        full_prompt = f"{instruction}\n\n{prompt}\n\n{task_details}:"
        logging.debug("Full Prompt: " + full_prompt)

        response_container['completion'] = openai.Completion.create(
            model="text-curie-001",  # Use a non-chat model
            prompt=full_prompt,
            max_tokens=10  # Adjust as needed
        )
    except Exception as e:
        response_container['exception'] = e

    if 'exception' in response_container:
        logging.error(f"Exception raised in API call: {response_container['exception']}")
    else:
        completion = response_container.get('completion')
        if completion:
            # Adjust how the response is parsed for non-chat completions
            response = completion.choices[0].text.strip()
            logging.debug("Response: " + response)
            if response == "1":
                return 1
            else:
                return 0
                