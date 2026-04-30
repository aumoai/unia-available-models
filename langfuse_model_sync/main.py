import requests
import re
from copy import deepcopy
import json

# -------- CONFIG --------
LANGFUSE_URL = "https://langfuse.dev.unia.aumo.live"
LANGFUSE_PUBLIC_KEY = ""
LANGFUSE_SECRET_KEY = ""
auth = (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
LANGFUSE_PAGE_LIMIT = 100

LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
UNIA_MODEL_FILE = "../models.json"

LITELLM_TO_LANFUSE_COST_NAMES = {
    "input_cost_per_token" : "input",
    "output_cost_per_token" : "output",
    "cache_read_input_token_cost" : "cache_read_input_tokens",
    "cache_creation_input_token_cost" : "input_cache_creation",
    "cache_creation_input_token_cost_above_1hr" : "input_cache_creation_1h",
}

BEDROCK_PROVIDER_NAMES = ["bedrock", "bedrock_converse"]
OCI_PROVIDER_NAMES = ["oci"]
# ------------------------


def get_langfuse_models():
    models = set()
    page = 1

    while True:
        r = requests.get(
            f"{LANGFUSE_URL}/api/public/models",
            params={"page": page, "limit": LANGFUSE_PAGE_LIMIT},
            auth=auth,
        )
        r.raise_for_status()
        data = r.json()
        models.update(m["modelName"] for m in data["data"])

        if page >= data["meta"]["totalPages"]:
            break

        page += 1

    return models

def get_pricing():
    r = requests.get(LITELLM_PRICING_URL)
    r.raise_for_status()
    return r.json()


def normalize_model(model):
    """
    bedrock/anthropic.claude-3-sonnet-20240229-v1:0
    ↓
    anthropic.claude-3-sonnet-20240229-v1:0
    """
    return model.split("/", 1)[1]


def extract_price(model):
    if not model:
        return None

    input_cost = model.get("input_cost_per_token")
    output_cost = model.get("output_cost_per_token")


    if input_cost is None or output_cost is None:
        return None

    return {
        "input": input_cost,
        "output": output_cost,
    }


def build_langfuse_payload(model_name, model_data):
    prices = {}

    for key, value in model_data.items():
        if "cost" not in key.lower():
            continue

        k = key.lower()

        if isinstance(value, dict):
            value = value.get("price")
        
        if value is None:
            continue
        if k in LITELLM_TO_LANFUSE_COST_NAMES:
            prices[LITELLM_TO_LANFUSE_COST_NAMES[k]] = float(value)
        else:
            print("Nome do token não presente em LITELLM_TO_LANFUSE_COST_NAMES, custos podem ser afetados!")
            prices[k] = float(value)

    if not prices:
        return None

    return {
        "modelName": model_name,
        "matchPattern": f"(?i)^{re.escape(model_name)}$",
        "unit": "TOKENS",
        "pricingTiers": [
            {
                "name": "Standard",
                "isDefault": True,
                "priority": 0,
                "conditions": [],
                "prices": prices
            }
        ]
    }

def create_langfuse_model(model_name: str, model_data: dict):

    payload = build_langfuse_payload(model_name,model_data)
    # print(payload)

    try:
        r = requests.post(
            f"{LANGFUSE_URL}/api/public/models",
            json=payload,
            auth=auth,
        )

    except:
        print("Erro ao criar o modelo no langfuse!")
        return None
    
    return payload["modelName"]

# def filter_bedrock_models(litellm_models: dict) -> dict:
#     bedrock_models = {}

#     for model_name, model_data in litellm_models.items():
#         provider = model_data.get("litellm_provider") or model_data.get("provider")

#         if provider == "bedrock" or provider == "bedrock_converse":
#             if not model_name.startswith("bedrock/"):
#                 # providers = ["nvidia", "mistral", "openai", "anthropic","qwen","meta"]
#                 # if any(p in model_name for p in providers):
#                 bedrock_models[model_name] = model_data

#     return bedrock_models

def filter_models_by_provider(litellm_models: dict, provider_names: list) -> dict:
    bedrock_models = {}

    for model_name, model_data in litellm_models.items():
        provider = model_data.get("litellm_provider") or model_data.get("provider")

        if provider in provider_names:
            if model_name.startswith(tuple(provider_names)):
                model_name = normalize_model(model_name)
                # providers = ["nvidia", "mistral", "openai", "anthropic","qwen","meta"]
                # if any(p in model_name for p in providers):
            bedrock_models[model_name] = model_data

    return bedrock_models

def get_models():
    r = requests.get(
        f"{LANGFUSE_URL}/api/public/models",
        auth=auth
    )
    r.raise_for_status()
    return r.json()["data"]


def delete_model(model_id):
    r = requests.delete(
        f"{LANGFUSE_URL}/api/public/models/{model_id}",
        auth=auth
    )
    if r.status_code not in (200, 204):
        print("Erro:", r.text)
        r.raise_for_status()

def find_unia_submodels(data):
    stack = [data]
    results = []

    while stack:
        current = stack.pop()

        if isinstance(current, dict):
            for k, v in current.items():
                if k == "submodels":
                    results.extend(v)
                stack.append(v)

        elif isinstance(current, list):
            stack.extend(current)

    return results

def main():

    print("Lendo JSON de Modelos...")
    # Pode melhorar pegando direto do Github mas não fiz
    with open(UNIA_MODEL_FILE, "r") as f:
        unia_models_json = json.load(f)
    unia_models_list = find_unia_submodels(unia_models_json)
    # print(json.dumps(unia_models_list, indent=2))

    print("Baixando pricing LiteLLM...")
    litellm_pricing = get_pricing()

    print("Filtrando Modelos Bedrock...")
    bedrock_models = filter_models_by_provider(litellm_pricing,BEDROCK_PROVIDER_NAMES)

    print("Filtrando Modelos OCI...")
    oci_models = filter_models_by_provider(litellm_pricing,OCI_PROVIDER_NAMES)
    # print(json.dumps(oci_models, indent=2))

    print("Filtrando Modelos Bedrock para modelos presentes no JSON de Modelos...")
    filtered_models = {
        model: data
        for models in (bedrock_models, oci_models)
        for model, data in models.items()
        if model in unia_models_list
    }
    #print(json.dumps(filtered_models, indent=2))
    
    print("Buscando modelos do Langfuse...")
    langfuse_models = get_langfuse_models()

    print("Removendo modelos já existentes no Langfuse...")
    updated_models = deepcopy(filtered_models)
    for model in filtered_models:

        # print(f"Model {model}")
        if model in langfuse_models:
            print(f"Modelo {model} já existe no Langfuse")
            updated_models.pop(model)
            continue
        price = extract_price(filtered_models[model])

        if not price:
            print("Sem preço:", model)
            updated_models.pop(model)
            continue


    # print(langfuse_models)
    # print(json.dumps(updated_models, indent=2))

    print("Criando Modelos no Langfuse...")
    if len(updated_models) > 0:
        models_created = []
        for model in updated_models:
            models_created.append(create_langfuse_model(model,updated_models[model]))
        
        for model in unia_models_list:
            if model not in models_created and model in updated_models:
                print(f"{model} was not possible to be created on Langfuse")

        print(f"Models created : ")
        for i, m in enumerate(sorted(models_created), 1):
            print(f"{i:2d}. {m}")
    else:
        print("Todos Modelos já estão na tabela de Modelos do Langfuse")



if __name__ == "__main__":
    main()