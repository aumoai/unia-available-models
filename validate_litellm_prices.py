#!/usr/bin/env python3
"""Valida que os custos dos models JSON da UNIA batem com o cost map do LiteLLM.

O LiteLLM (`model_prices_and_context_window.json`) e a fonte da verdade. Para cada
submodelo ativo, compara `input_cost_per_token` e `output_cost_per_token`.

Uso:
    python validate_litellm_prices.py                      # models.json
    python validate_litellm_prices.py models_aws.json models_bees.json
    python validate_litellm_prices.py --litellm-file litellm.json   # offline
    python validate_litellm_prices.py --strict             # nao encontrado tambem falha

Saida: 0 tudo igual; 1 divergencia (ou nao encontrado com --strict); 2 erro de uso/rede.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
DEFAULT_FILES = ["models.json"]

# base_model_name (UNIA) -> valores de `litellm_provider` aceitos no cost map.
LITELLM_PROVIDERS: dict[str, set[str]] = {
    "openai": {"openai", "text-completion-openai"},
    "bedrock": {"bedrock", "bedrock_converse"},
    "azure_ai": {"azure_ai"},
    "azure": {"azure", "azure_text"},
    "gemini": {"gemini"},
    "vertex_ai": {"vertex_ai", "vertex_ai-language-models", "vertex_ai-chat-models"},
    "fireworks_ai": {"fireworks_ai"},
    "oci": {"oci"},
    "anthropic": {"anthropic"},
}
# Providers proprios da UNIA/AUMO: nao existem no LiteLLM, ficam fora da validacao.
DEFAULT_SKIP_PROVIDERS = {"mock", "aumo"}
DEFAULT_SKIP_DISPLAY_NAMES = {"AUMO Neuron"}

REL_TOL = 1e-9
COST_FIELDS = ("input_cost_per_token", "output_cost_per_token")


@dataclass
class Result:
    file: str
    provider: str
    submodel: str
    status: str  # ok | mismatch | not_found | no_price
    litellm_key: str | None = None
    detail: str = ""


def load_json(source: str) -> dict:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as response:  # noqa: S310
            return json.load(response)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def build_index(litellm: dict) -> dict[str, dict[str, tuple[str, dict]]]:
    """litellm_provider -> {nome minusculo -> (chave original, entrada)}.

    Cada entrada e indexada pela chave inteira e pela chave sem o prefixo
    `provider/` (so quando sobra um nome simples: chaves regionais como
    `bedrock/us-gov-west-1/x` tem preco proprio e nao podem valer para `x`).
    Fireworks usa o ultimo segmento de `accounts/fireworks/models/x`.
    """
    index: dict[str, dict[str, tuple[str, dict]]] = {}
    for key, entry in litellm.items():
        if not isinstance(entry, dict):
            continue
        provider = entry.get("litellm_provider")
        if not provider:
            continue
        lowered = key.lower()
        names = {lowered}
        if "/" in lowered:
            rest = lowered.split("/", 1)[1]
            if "/" not in rest:
                names.add(rest)
            elif provider == "fireworks_ai":
                names.add(lowered.rsplit("/", 1)[-1])
        bucket = index.setdefault(provider, {})
        for name in names:
            # A chave exata ganha de aliases derivados.
            if name == lowered or name not in bucket:
                bucket[name] = (key, entry)
    return index


def resolve(
    index: dict[str, dict[str, tuple[str, dict]]], providers: set[str], submodel: str
) -> tuple[str, dict] | None:
    name = submodel.lower()
    for provider in sorted(providers):
        hit = index.get(provider, {}).get(name)
        if hit:
            return hit
    return None


def same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=0.0)


def validate_file(
    path: str,
    index: dict[str, dict[str, tuple[str, dict]]],
    skip_providers: set[str],
    skip_display_names: set[str],
    include_inactive: bool,
) -> list[Result]:
    data = load_json(path)
    results: list[Result] = []
    for provider in data["providers"]:
        base = provider["base_model_name"]
        if base.lower() in skip_providers or provider.get("displayName") in skip_display_names:
            continue
        litellm_providers = LITELLM_PROVIDERS.get(base)
        for model in provider["models"]:
            if not include_inactive and model.get("status") != "active":
                continue
            for submodel in model["submodels"]:
                if litellm_providers is None:
                    results.append(
                        Result(path, base, submodel, "not_found", detail="provider sem mapeamento")
                    )
                    continue
                hit = resolve(index, litellm_providers, submodel)
                if hit is None:
                    results.append(Result(path, base, submodel, "not_found"))
                    continue
                key, entry = hit
                if any(entry.get(field) is None for field in COST_FIELDS):
                    results.append(Result(path, base, submodel, "no_price", key))
                    continue
                diffs = [
                    f"{field.split('_')[0]}: unia={model.get(field)!r} litellm={entry[field]!r}"
                    for field in COST_FIELDS
                    if model.get(field) is None or not same(float(model[field]), float(entry[field]))
                ]
                results.append(
                    Result(
                        path, base, submodel, "mismatch" if diffs else "ok", key, "; ".join(diffs)
                    )
                )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="*", help=f"models JSON (default: {DEFAULT_FILES})")
    parser.add_argument("--litellm-file", help="cost map local em vez de baixar do GitHub")
    parser.add_argument("--strict", action="store_true", help="modelo nao encontrado tambem falha")
    parser.add_argument("--include-inactive", action="store_true", help="valida status != active")
    parser.add_argument("--skip-provider", action="append", default=[], help="base_model_name")
    parser.add_argument("--json", action="store_true", help="saida em JSON")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    files = args.files or [str(base_dir / name) for name in DEFAULT_FILES]
    try:
        litellm = load_json(args.litellm_file or LITELLM_PRICING_URL)
    except OSError as exc:
        print(f"erro ao carregar o cost map do LiteLLM: {exc}", file=sys.stderr)
        return 2

    index = build_index(litellm)
    skip = DEFAULT_SKIP_PROVIDERS | {p.lower() for p in args.skip_provider}
    results: list[Result] = []
    for path in files:
        results.extend(
            validate_file(path, index, skip, DEFAULT_SKIP_DISPLAY_NAMES, args.include_inactive)
        )

    counts = {s: sum(r.status == s for r in results) for s in ("ok", "mismatch", "not_found", "no_price")}
    if args.json:
        print(json.dumps({"counts": counts, "results": [r.__dict__ for r in results]}, indent=2))
    else:
        for label, status in (
            ("DIVERGENTE", "mismatch"),
            ("SEM PRECO NO LITELLM", "no_price"),
            ("NAO ENCONTRADO NO LITELLM", "not_found"),
        ):
            rows = [r for r in results if r.status == status]
            if not rows:
                continue
            print(f"\n{label} ({len(rows)})")
            for r in rows:
                extra = f"  [{r.litellm_key}]" if r.litellm_key else ""
                detail = f"  {r.detail}" if r.detail else ""
                print(f"  {Path(r.file).name}  {r.provider}/{r.submodel}{extra}{detail}")
        print(
            f"\n{counts['ok']} ok, {counts['mismatch']} divergentes, "
            f"{counts['not_found']} nao encontrados, {counts['no_price']} sem preco no litellm"
        )

    failed = counts["mismatch"] > 0 or (args.strict and counts["not_found"] > 0)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
