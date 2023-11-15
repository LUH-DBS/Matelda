import logging
import os
import threading
from time import sleep
import time
import openai


def get_foundation_model_prediction(tables_tuples_dict, key):
    table_id = key[0]
    header = tables_tuples_dict[table_id]["header"]
    tuple = tables_tuples_dict[table_id]["tuples"][key[2]]
    ground_truth = tables_tuples_dict[table_id]["clean"][key[2]]
    attribute_idx = key[1]
    attribute_val = header[attribute_idx]
    value = tuple[attribute_idx]
    value_gt = ground_truth[attribute_idx]
    attribute_value_str = ""
    for col_idx, col in enumerate(header):
        attribute_value_str += col + ": " + tuple[col_idx] + ", "
    prompt = f"tuple: {attribute_value_str}, specified attribute: {attribute_val}"
    label = ask_model(prompt)
    logging.info("Asking Foundation Model: prompt: " + prompt + " GPT label: " + str(label) + " Dirty Value " + value + " Clean Value " + value_gt)
    sleep(1)
    return label

def few_shot_prediction(tables_tuples_dict, keys, user_samples_dict):
    try:
        logging.info("FEW SHOT PREDICTION")
        few_shot_strings = "Example Tuples:\n"
        for sample in user_samples_dict.keys():
            table_id = sample[0]
            header = tables_tuples_dict[table_id]["header"]
            tuple = tables_tuples_dict[table_id]["tuples"][sample[2]]
            user_label = user_samples_dict[sample]
            attribute_idx = sample[1]
            attribute_val = header[attribute_idx]
            attribute_value_str = ""
            for col_idx, col in enumerate(header):
                attribute_value_str += col + ": " + tuple[col_idx] + ", "
            sample_info = f"tuple: {attribute_value_str} specified attribute: {attribute_val}, user label: {user_label}"
            few_shot_strings += sample_info + "\n"
        test_strings = "Test Tuples:\n"
        test_values_dict = {"value": [], "value_gt": []}
        for key in keys:
            table_id = key[0]
            header = tables_tuples_dict[table_id]["header"]
            tuple = tables_tuples_dict[table_id]["tuples"][key[2]]
            ground_truth = tables_tuples_dict[table_id]["clean"][key[2]]
            attribute_idx = key[1]
            attribute_val = header[attribute_idx]
            test_values_dict["value"].append(tuple[attribute_idx])
            test_values_dict["value_gt"].append(ground_truth[attribute_idx])
            attribute_value_str = ""
            for col_idx, col in enumerate(header):
                attribute_value_str += col + ": " + tuple[col_idx] + ", "
            test_strings += f"tuple: {attribute_value_str} specified attribute: {attribute_val}\n"
        prompt = few_shot_strings + test_strings
        test_labels = ask_model_few_shot_comp(prompt, len(keys))
        logging.info("Finished few shot prediction")
        logging.info("Few shot prompt: " + prompt)
        results = ""
        for i in range(len(test_labels)):
            results += "Foundation Model label: " + str(test_labels[i]) + " Dirty Value " + test_values_dict["value"][i] + " Clean Value " + test_values_dict["value_gt"][i] + "\n"
        logging.info(results)
    except Exception as e:
        logging.error("Error in few_shot_prediction: " + str(e))
    return test_labels
   
def api_call_comp(prompt, response_container):
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        instruction = "Given tuples with multiple attribute-value pairs, evaluate if the specified attributes' values are correct. For each attribute in question, respond only with 'Yes' if the value is correct or 'No' if it is erroneous. Provide your responses separated by commas, like 'Yes, No'."
        
        # Assumed format of prompt: "Examples:\n[Example Tuples]\n\nTest Tuples:\n[Test Tuples]"
        full_prompt = f"{instruction}\n\n{prompt}\n\nResponse:"
        
        response_container['completion'] = openai.Completion.create(
            model="text-curie-001",  # Use a non-chat model
            prompt=full_prompt,
            max_tokens=150  # Adjust as needed
        )
    except Exception as e:
        response_container['exception'] = e


def api_call(prompt, response_container):
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response_container['completion'] = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "Given tuples with multiple attribute-value pairs and a user label indicating the correctness of specific attributes, evaluate two test tuples. For each attribute in question, respond with 'Yes' or 'No' to signify if their values are erroneous. Provide answers for each attribute in the test tuples, separated by commas, like 'Yes, No, Yes'."},
                {"role": "user", "content": prompt},
            ]
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
        thread = threading.Thread(target=api_call_comp, args=(prompt, response_container))
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
                    if len(responses) >= n_shots:
                        for i in range(n_shots):
                            if responses[i].strip() == "Yes":
                                test_labels.append(1)
                            else:
                                test_labels.append(0)
                        logging.info("Got Answer from the Model")
                        return test_labels

        time.sleep(retry_delay)
        retry_delay *= backoff_factor

    logging.error("Error in ask_model_few_shot after maximum retries")
    return [0 for i in range(n_shots)]

def ask_model(prompt):
    try:
        response_container = {}
        openai.api_key = os.getenv("OPENAI_API_KEY")
        instruction = "Given tuples with multiple attribute-value pairs, evaluate if the specified attributes' values are correct. For each attribute in question, respond only with 'Yes' if the value is correct or 'No' if it is erroneous. Provide your responses separated by commas, like 'Yes, No'."
        
        # Assumed format of prompt: "Examples:\n[Example Tuples]\n\nTest Tuples:\n[Test Tuples]"
        full_prompt = f"{instruction}\n\n{prompt}\n\nResponse:"
        
        response_container['completion'] = openai.Completion.create(
            model="text-curie-001",  # Use a non-chat model
            prompt=full_prompt,
            max_tokens=150  # Adjust as needed
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
            if response == "Yes":
                return 1
            else:
                return 0
                