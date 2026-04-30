# Langfuse Model Sync

If any model is added to any "models.json", be sure to run this program to sync Langfuse model table with the LiteLLM model cost map table.

The script only adds models that do not exist on Langfuse. An update feature to sync costs of already existing models could be needed!

## Attention

As of 30/04/2026, no official support for LiteLLM was found that enable ingesting costs of unknown models to Langfuse, the "main.py" script is just a workaround for showing correct model costs on Langfuse. It may contain bugs!