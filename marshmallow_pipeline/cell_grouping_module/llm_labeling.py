import logging
import os
from time import sleep
import openai


def get_foundation_nodel_prediction(tables_tuples_dict, key):
    table_id = key[0]
    header = tables_tuples_dict[table_id]["header"]
    tuple = tables_tuples_dict[table_id]["tuples"][key[2]]
    attribute = key[1]
    attribute_value_str = ""
    for col_idx, col in enumerate(header):
        attribute_value_str += col + ": " + tuple[col_idx] + ", "
    prompt = f"tuple: {attribute_value_str}, specified attribute: {attribute}"
    label = ask_model(prompt)
    sleep(0.5)
    return label



def ask_model(prompt):
  try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "I give you a tuple (A (attribute A): v (value v) and specify one attribute. You specify whether the value for the specified attribut in that tuple is errouneous or not with Yes or No."},
        {"role": "user", "content": prompt},
      ]
    )
    answer = completion.choices[0].message["content"][0:3]
    label = 1 if answer == "Yes" else 0
    logging.info("Got Answer from GPT-3")
  except:
    label = 0
    logging.error("Error in ask_model")
  return label