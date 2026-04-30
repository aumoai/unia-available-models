# unia-available-models
Repository with current models supported by UNIA.

If any model is added to any JSON, be sure to run the script present in [langfuse_model_sync](./langfuse_model_sync/main.py).

If the model is custom (which means is not present on LiteLLM cost map), it's needed to add support for it in [langfuse_model_sync](./langfuse_model_sync/main.py).