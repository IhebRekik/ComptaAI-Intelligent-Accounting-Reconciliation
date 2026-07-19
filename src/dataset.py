#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataset.py
==========

Module de gestion des données du projet ComptaAI.

Responsabilités :
    - Lecture des fichiers JSONL (train / validation / test)
    - Validation du contenu (champs obligatoires : `instruction`, `response`)
    - Construction des prompts d'entraînement via `prompt.build_training_prompt`
    - Tokenisation avec le tokenizer de Qwen
    - Création des objets `Dataset` de Hugging Face
    - Fonctions utilitaires : statistiques, aperçu, mélange, filtrage par
      longueur, export des datasets

Ce module est directement utilisé par `train.py` via les fonctions
`load_jsonl_dataset` et `build_prompt_dataset`. Sa signature ne doit pas
changer afin de rester compatible avec `train.py`, qui est considéré
comme terminé.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset

from prompt import build_training_prompt

logger = logging.getLogger("comptaai.dataset")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                           datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

REQUIRED_FIELDS = ("instruction", "response")


# ---------------------------------------------------------------------------
# Chargement et validation
# ---------------------------------------------------------------------------

def load_jsonl_dataset(path: str, min_response_length: int = 1) -> List[Dict[str, Any]]:
    """Charge un fichier JSONL et valide chaque exemple.

    Chaque ligne du fichier doit être un objet JSON contenant au minimum
    les champs `instruction` et `response`. Les lignes invalides (JSON
    malformé, champs manquants, réponse trop courte) sont ignorées et
    signalées dans les logs plutôt que de faire échouer le chargement.

    Args:
        path: Chemin vers le fichier .jsonl.
        min_response_length: Longueur minimale (en caractères, après
            nettoyage) exigée pour la réponse d'un exemple.

    Returns:
        Une liste de dictionnaires validés, chacun contenant au minimum
        les clés `instruction` et `response`.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")

    examples: List[Dict[str, Any]] = []
    skipped = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"[{path}:{line_num}] Ligne JSON invalide, ignorée.")
                skipped += 1
                continue

            if not _is_valid_record(record, min_response_length):
                logger.warning(f"[{path}:{line_num}] Exemple invalide, ignoré.")
                skipped += 1
                continue

            examples.append({
                "instruction": str(record["instruction"]).strip(),
                "response": str(record["response"]).strip(),
            })

    logger.info(
        f"{path} -> {len(examples)} exemples valides chargés "
        f"({skipped} ignorés)."
    )
    return examples


def _is_valid_record(record: Dict[str, Any], min_response_length: int) -> bool:
    """Vérifie qu'un enregistrement contient bien les champs obligatoires
    et que la réponse respecte la longueur minimale attendue."""
    if not isinstance(record, dict):
        return False

    for field in REQUIRED_FIELDS:
        if field not in record or not isinstance(record[field], str):
            return False

    if len(record["response"].strip()) < min_response_length:
        return False

    if len(record["instruction"].strip()) == 0:
        return False

    return True


# ---------------------------------------------------------------------------
# Construction du dataset de prompts + tokenisation
# ---------------------------------------------------------------------------

def build_prompt_dataset(
    raw_examples: List[Dict[str, Any]],
    tokenizer,
    config: Dict[str, Any],
) -> Dataset:
    """Transforme une liste d'exemples bruts en `Dataset` Hugging Face
    prêt pour l'entraînement avec `SFTTrainer`.

    Pour chaque exemple, le prompt complet (instruction + réponse) est
    construit via `prompt.build_training_prompt`, puis tokenisé avec le
    tokenizer fourni. Un champ `labels` (copie des `input_ids`) est ajouté
    pour une compatibilité maximale avec les différentes versions de TRL.

    Args:
        raw_examples: Liste d'exemples validés (`instruction`, `response`).
        tokenizer: Tokenizer Hugging Face associé au modèle Qwen.
        config: Configuration globale du projet (issue de qwen_lora.yaml).

    Returns:
        Un `datasets.Dataset` contenant les colonnes `text`, `input_ids`,
        `attention_mask` et `labels`.
    """
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    max_seq_length = model_cfg.get("max_seq_length", 2048)

    if data_cfg.get("shuffle", True):
        raw_examples = shuffle_dataset(raw_examples, seed=config.get("training", {}).get("seed", 42))

    max_samples = data_cfg.get("max_samples")
    if max_samples:
        raw_examples = raw_examples[:max_samples]

    texts = [
        build_training_prompt(ex["instruction"], ex["response"])
        for ex in raw_examples
    ]

    dataset = Dataset.from_dict({"text": texts})

    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    dataset = dataset.map(_tokenize, batched=True, desc="Tokenisation des exemples")

    dataset = filter_by_max_length(dataset, max_seq_length)

    return dataset


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def dataset_statistics(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcule des statistiques descriptives sur un jeu de données brut :
    nombre d'exemples, longueurs moyennes/min/max des instructions et
    des réponses (en caractères)."""
    if not examples:
        return {"count": 0}

    instr_lengths = [len(ex["instruction"]) for ex in examples]
    resp_lengths = [len(ex["response"]) for ex in examples]

    stats = {
        "count": len(examples),
        "instruction_length": {
            "min": min(instr_lengths),
            "max": max(instr_lengths),
            "avg": sum(instr_lengths) / len(instr_lengths),
        },
        "response_length": {
            "min": min(resp_lengths),
            "max": max(resp_lengths),
            "avg": sum(resp_lengths) / len(resp_lengths),
        },
    }

    logger.info(f"Statistiques du dataset : {stats}")
    return stats


def preview_examples(examples: List[Dict[str, Any]], n: int = 3) -> None:
    """Affiche un aperçu lisible des `n` premiers exemples du dataset,
    utile pour vérifier rapidement le contenu avant l'entraînement."""
    for i, ex in enumerate(examples[:n]):
        logger.info(f"--- Exemple {i + 1} ---")
        logger.info(f"Instruction : {ex['instruction']}")
        logger.info(f"Réponse     : {ex['response']}")


def shuffle_dataset(
    examples: List[Dict[str, Any]], seed: Optional[int] = 42
) -> List[Dict[str, Any]]:
    """Mélange aléatoirement les exemples du dataset (copie, non
    destructif) en utilisant une graine pour la reproductibilité."""
    shuffled = examples.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def filter_by_max_length(dataset: Dataset, max_length: int) -> Dataset:
    """Filtre les exemples déjà tokenisés dont le nombre de tokens dépasse
    `max_length`, afin d'éviter les troncatures silencieuses en cours
    d'entraînement."""
    before = len(dataset)
    dataset = dataset.filter(lambda ex: len(ex["input_ids"]) <= max_length)
    after = len(dataset)

    if before != after:
        logger.info(f"Filtrage par longueur max ({max_length}) : {before} -> {after} exemples.")

    return dataset


def export_dataset(examples: List[Dict[str, Any]], output_path: str) -> None:
    """Exporte une liste d'exemples au format JSONL vers `output_path`,
    utile pour sauvegarder un sous-échantillon ou un dataset nettoyé."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"{len(examples)} exemples exportés vers {output_path}")
