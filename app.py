from flask import Flask, request, jsonify
import json
import os
import hashlib

from concurrent.futures import ThreadPoolExecutor

from google.cloud import dialogflow_v2 as dialogflow
from google.oauth2 import service_account

from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

executor = ThreadPoolExecutor(max_workers=1)

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m", use_cache=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

app = Flask(__name__)

project_id = "bloom-560m-oxcs"  # Substitua pelo ID do seu projeto no GCP

# Carregue suas credenciais
credentials = service_account.Credentials.from_service_account_file(
    "./bloom-560m-oxcs-0d0eb16ef2b6.json")

# Crie um cliente Dialogflow
intents_client = dialogflow.IntentsClient(credentials=credentials)

# Configure a localização do agente e o ID do projeto
agent_path = f"projects/{project_id}/agent"


def generate(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")
    sample = model.generate(**input_ids, max_length=256, temperature=0.1)

    return tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^”'", "\n\n\n"])


def create_intent(pergunta):
    try:
        # Configure training phrases and text messages
        training_phrase = dialogflow.types.Intent.TrainingPhrase(parts=[
            dialogflow.types.Intent.TrainingPhrase.Part(text=pergunta)
        ])

        text = dialogflow.types.Intent.Message.Text(text=[generate(pergunta)])

        # Configure the intent
        intent = dialogflow.types.Intent(
            display_name=hashlib.sha1(pergunta.encode("utf-8")).hexdigest(),
            training_phrases=[training_phrase],
            messages=[dialogflow.types.Intent.Message(text=text)],
        )

        # Create the request
        create_intent_request = dialogflow.types.CreateIntentRequest(
            parent=agent_path,
            intent=intent,
        )

        # Send the request and display the result
        response = intents_client.create_intent(request=create_intent_request)
        print(f"Intent created: {response.name}")
    except Exception as error:
        print(f"Error updating intent: {error}")


@app.route("/", methods=["POST"])
def webhook():
    req_data = request.get_json()

    print("Query", req_data["queryResult"]["queryText"])
    # print("Query", req_data["queryResult"])

    executor.submit(create_intent, req_data["queryResult"]["queryText"])

    res = {
        "fulfillmentText": "I'm learning yet",
    }

    return jsonify(res)


if __name__ == "__main__":
    app.run()
