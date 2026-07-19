#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluation.py
=============

Module d'évaluation du modèle ComptaAI après entraînement.

Permet de :
    - Charger soit le modèle fusionné, soit le modèle de base + adaptateurs
      LoRA
    - Exécuter l'évaluation sur le jeu de test (`dataset/test.jsonl`)
    - Calculer différentes métriques de performance (loss, perplexité)
    - Sauvegarder les résultats pour analyse (JSON + CSV)

Utilisation en ligne de commande :
    python src/evaluation.py --config configs/qwen_lora.yaml --mode lora
    python src/evaluation.py --config configs/qwen_lora.yaml --mode merged
"""

import argparse
import math
from pathlib import Path
from typing import Any, Dict

from dataset import load_jsonl_dataset, build_prompt_dataset
from utils import load_yaml_config, setup_logger, save_json, save_csv, generate_timestamp

logger = setup_logger("comptaai.evaluation")


# ---------------------------------------------------------------------------
# Arguments CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évaluation du modèle ComptaAI.")
    parser.add_argument("--config", type=str, default="configs/qwen_lora.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "merged"],
        default="lora",
        help="Charger le modèle avec les adaptateurs LoRA ou le modèle fusionné.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------

def load_model_for_evaluation(config: Dict[str, Any], mode: str):
    """Charge le modèle à évaluer, soit sous forme LoRA (modèle de base +
    adaptateurs), soit sous forme fusionnée, selon `mode`."""
    from unsloth import FastLanguageModel

    dirs = config["directories"]
    model_cfg = config["model"]

    if mode == "merged":
        model_path = dirs["merged_dir"]
        logger.info(f"Chargement du modèle fusionné depuis : {model_path}")
    else:
        model_path = dirs["lora_dir"]
        logger.info(f"Chargement du modèle de base + adaptateurs LoRA depuis : {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        dtype=model_cfg.get("dtype", None),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, tokenizer, config: Dict[str, Any]) -> Dict[str, Any]:
    """Exécute l'évaluation du modèle sur le jeu de test et calcule la
    perte moyenne ainsi que la perplexité associée."""
    import torch
    from transformers import Trainer, TrainingArguments

    data_cfg = config["data"]
    test_raw = load_jsonl_dataset(data_cfg["test_path"])
    test_dataset = build_prompt_dataset(test_raw, tokenizer, config)

    logger.info(f"Évaluation sur {len(test_dataset)} exemples de test.")

    training_args = TrainingArguments(
        output_dir="outputs/eval_tmp",
        per_device_eval_batch_size=config["training"].get("eval_batch_size", 2),
        report_to=[],
        do_train=False,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    with torch.no_grad():
        metrics = trainer.evaluate()

    eval_loss = metrics.get("eval_loss")
    if eval_loss is not None:
        metrics["perplexity"] = math.exp(eval_loss)

    logger.info(f"Résultats d'évaluation : {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    model, tokenizer = load_model_for_evaluation(config, args.mode)
    metrics = evaluate_model(model, tokenizer, config)

    output_dir = config["directories"].get("output_dir", "outputs")
    timestamp = generate_timestamp()
    json_path = str(Path(output_dir) / f"evaluation_{args.mode}_{timestamp}.json")
    csv_path = str(Path(output_dir) / f"evaluation_{args.mode}_{timestamp}.csv")

    save_json(metrics, json_path)
    save_csv([metrics], csv_path)

    logger.info(f"Résultats sauvegardés dans {json_path} et {csv_path}")


if __name__ == "__main__":
    main()
