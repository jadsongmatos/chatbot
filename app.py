from flask import Flask, request, jsonify
import json
import os
import hashlib

from google.cloud import dialogflow_v2 as dialogflow
from google.oauth2 import service_account

from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m", use_cache=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

app = Flask(__name__)

project_id = "bloom-560m-oxcs"  # Substitua pelo ID do seu projeto no GCP

# Carregue suas credenciais
credentials = service_account.Credentials.from_service_account_info(
    {
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDIbI73s5cVrbtt\nyVRlEXRa7yEJeDNRDTFOCv00osab3eA0Sr30UaKa/QRFcvIc5SU+sHA72IpeRvYT\nRGQc94JuiJ7yCrk94IIeC64wnwOep3Pj4dTmuE2cbz8DRUhZNnUWYFY6AOJ2h/qE\nFXOlK/DusEFOdnsKdE47p3CHLikKn36ivkpMPDhvoVvjPND3LDutKUJ20+6VtDE6\nR0lGqVUQlcbcxaOYLRD5wh7B5e8Q0yOYSq7Pu3cUVXZpa99In0DLsFyPKRgIsdSa\nqZcH6ck3ZLt78TwsUFuMRCnunxNUerCNP2djEdx2ae6j+ISsco4RNG3XfnoGwwK5\n0qDYZ2x/AgMBAAECggEAALAdnVUMQGV626DXg10d1nJAI0eu7zpsl34IQUjzwDUf\ncWRTYIBzDTWYfL+hZNktx/VZ5rn1sLgnXRRim8TYc7khsXKKnLC3LraVRyoVSw1Z\nF+WnOr9UNIGq4pE3W9vDIZSTtbz8laY1Ym1RVRNmTy2dbfQb4tQqAjfGUz5HoyXr\nLRzahxnQhn26LF7FJbEU+Spz/WD1y/tmknyKmt1+TDsgH+dVwiieemw9KUU55OiL\n2fDO1QIcyN73b8aDi7sSSqmX5wKCfkZwoQPylr1nefae19+DfDbwWLs3RyBMNtqc\ncsJlTfVSbrOY385akmys8HRVvYgYlt80mo2kRkTbSQKBgQDn+g5l5GslpoUlKKOu\nqrtEqN8Euob1VddF3vjdv7IQGn8l/WMa0VeWKrPVkT9dDbuXKxn2obzdRZ8dLo5s\nrVsfu3fO+RcEi2x9ibTMuUFuNFvu0Fx7MPmFwXYQhe/UXWAU9p5vLfhvzr4zV2Yr\nSn0fsUTmHccqTM2DtJXUjKgOpwKBgQDdLgHFuyyXU3PpcL5o5gS31YIwRVUaHE7q\nwsmVOPPF+S0JH1CA+RjulE4FrMUb+MMyEr6OfU3tksjTzsJJg9rxADZ2PjvRG+Rb\nQVZ9Ekynqu93U7QbWU+txK6I9Uw2xl5JIHQZyiNGChMnAK+2y9+0k47jYiwUBHkr\ntAElKYKGaQKBgQCoFTllftJcH4IN4JppLvAt2aZuiLDNBvvKdrsgAYwFuw0x+51Z\niyHJfvt63Zlp1U99Qw+28o4kThPyUw+Dk0CZh35SC44wDs33UFYemITeUzXhnjvE\nT0IdNoZThpi16Up7Jg66RDbk3IpYLRWsfHaZBhsEGcN71NZ8fdaqgZgrtQKBgEVd\nLU8ekTBwCpT2N7DcGjSw7+BFjhffMtyq898pekHnEDmhKXUZKbmApytecH6COG0W\ntc4n++16sCLE3+eAQ4R7RZk1kTrWHXqE9iPJPmMC15C7nmtJShS77uSi0SEIev9B\n/0crPn6zoCgGLhUMwP53nEMxQZORh5KuJAHMj9OpAoGAaTXvLuwGs3yVOLtz6qd1\ns9tGqadhYciNRDsS4/TclOW3FwnKDRv0ChJR2JXotch2fEMUV+Db3bKZafENX/Ic\nbr/L64zFCBuZuWVG4miZeTZ9jmYMdVXxBmOR6eBrPs/u0IjIdEC0At1CTj8xcwBc\ngzGqpVqseFAzde7pS9cRtdw=\n-----END PRIVATE KEY-----\n",
        "client_email": "bloom-560m@bloom-560m-oxcs.iam.gserviceaccount.com",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
)

# Crie um cliente Dialogflow
intents_client = dialogflow.IntentsClient(credentials=credentials)

# Configure a localização do agente e o ID do projeto
agent_path = f"projects/{project_id}/agent"


def generate(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")
    sample = model.generate(**input_ids, max_length=256, temperature=0.1,)

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

    create_intent(req_data["queryResult"]["queryText"])

    res = {
        "fulfillmentText": "I'm learning yet",
    }

    return jsonify(res)


if __name__ == "__main__":
    app.run()
