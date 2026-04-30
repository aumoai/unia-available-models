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
AZURE_PROVIDER_NAMES = ["azure","azure_ai","azure_text"]
ANTHROPIC_PROVIDER_NAMES = ["anthropic"]
GOOGLE_PROVIDER_NAMES = ["gemini"]
CURRENT_PROVIDERS = ["Azure","Oracle","Bedrock"]

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
    if '/' in model:
        return model.split("/", 1)[1]
    return model


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
            print(f"Nome do token não presente em LITELLM_TO_LANFUSE_COST_NAMES, custos podem ser afetados! modelo : {model_name}")
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


def filter_models_by_provider(litellm_models: dict, provider_names: list) -> dict:
    models = {}

    for model_name, model_data in litellm_models.items():
        provider = model_data.get("litellm_provider") or model_data.get("provider")

        if provider in provider_names:
            if model_name.startswith(tuple(provider_names)):
                model_name = normalize_model(model_name)
                # providers = ["nvidia", "mistral", "openai", "anthropic","qwen","meta"]
                # if any(p in model_name for p in providers):
            models[model_name] = model_data

    return models

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
    results = {}
    for provider in data["providers"]:
        model_list = []
        for model in provider["models"]:
            if "status" in model and model["status"] == "active":
                model_list.extend(submodel.lower() for submodel in model["submodels"])
        results[provider["name"]] = {provider["base_model_name"] : model_list}
    return results


def main():

    print("Lendo JSON de Modelos...")
    # Pode melhorar pegando direto do Github mas não fiz
    with open(UNIA_MODEL_FILE, "r") as f:
        unia_models_json = json.load(f)
    # print(json.dumps(unia_models_json, indent=2))
    unia_models_dict = find_unia_submodels(unia_models_json)
    #print(json.dumps(unia_models_dict, indent=2))

    print("Baixando pricing LiteLLM...")
    litellm_pricing = get_pricing()

    print("Filtrando Modelos Bedrock...")
    bedrock_models = filter_models_by_provider(litellm_pricing,BEDROCK_PROVIDER_NAMES)

    print("Filtrando Modelos OCI...")
    oci_models = filter_models_by_provider(litellm_pricing,OCI_PROVIDER_NAMES)
    # print(json.dumps(oci_models, indent=2))

    print("Filtrando Modelos Azure...")
    azure_models = filter_models_by_provider(litellm_pricing,AZURE_PROVIDER_NAMES)

    print("Filtrando Modelos Anthropic...")
    anthropic_models = filter_models_by_provider(litellm_pricing,ANTHROPIC_PROVIDER_NAMES)

    print("Filtrando Modelos Google...")
    google_models = filter_models_by_provider(litellm_pricing,GOOGLE_PROVIDER_NAMES)
    # print(json.dumps(oci_models, indent=2))

    print("Filtrando Modelos para modelos presentes no JSON de Modelos...")
    all_unia_models = [
        model
        for group in unia_models_dict.values()
        for models in group.values()
        for model in models
    ]
    all_unia_providers = [
        provider
        for group in unia_models_dict.values()
        for provider in group.keys()
    ]

    filtered_models = {
        model: data
        for models in (bedrock_models, oci_models, azure_models, anthropic_models, google_models)
        for model, data in models.items()
        if model in all_unia_models
        if data["litellm_provider"] in all_unia_providers
    }

    ### É possível algum modelo que existe porém não no provedor correto que ofertamos, portanto:
    true_filtered_models = deepcopy(filtered_models)

    for model, data in filtered_models.items():
        provider = data["litellm_provider"]
        for provider_group in unia_models_dict.values():
            for provider, models in provider_group.items():
                if data["litellm_provider"] == provider and model not in models:
                    true_filtered_models.pop(model, None)


    #print(json.dumps(true_filtered_models, indent=2))
    
    print("Buscando modelos do Langfuse...")
    langfuse_models = get_langfuse_models()

    print("Removendo modelos já existentes no Langfuse...")
    updated_models = deepcopy(true_filtered_models)
    for model in true_filtered_models:

        # print(f"Model {model}")
        if model in langfuse_models:
            print(f"Modelo {model} já existe no Langfuse")
            updated_models.pop(model)
            continue
        price = extract_price(true_filtered_models[model])

        if not price:
            print("Sem preço:", model)
            updated_models.pop(model)
            continue

    for model in all_unia_models:
        if model not in langfuse_models:
            if model not in filtered_models:
                print(f"Não será possível criar o modelo: {model} automaticamente no Langfuse,o preço no LiteLLM não existe ou provedor não foi incluso, provedores atuais : {CURRENT_PROVIDERS}!")
    # print(langfuse_models)
    # print(json.dumps(updated_models, indent=2))

    print("Criando Modelos no Langfuse...")
    if len(updated_models) > 0:
        models_created = []
        for model in updated_models:
            models_created.append(create_langfuse_model(model,updated_models[model]))
        
        for model in all_unia_models:
            if model not in models_created and model in updated_models:
                print(f"Não foi possível criar o modelo: {model} no Langfuse")

        print(f"Models created : ")
        for i, m in enumerate(sorted(models_created), 1):
            print(f"{i:2d}. {m}")
    else:
        print("Todos Modelos já estão na tabela de Modelos do Langfuse")



if __name__ == "__main__":
    main()