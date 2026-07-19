#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py
========

Script principal d'entraînement du projet ComptaAI.

Ce script orchestre le fine-tuning LoRA (Low-Rank Adaptation) du modèle
Qwen 3 afin de le spécialiser dans le domaine de la comptabilité
(lettrage comptable, analyse d'écritures, assistance aux comptables,
compréhension de documents financiers, etc.).

Étapes réalisées par ce script :
    1. Chargement de la configuration (configs/qwen_lora.yaml)
    2. Initialisation de l'environnement (seed, détection matériel)
    3. Création des répertoires du projet
    4. Chargement des jeux de données (train / val / test)
    5. Chargement du tokenizer et du modèle de base (Unsloth, 4-bit)
    6. Application de la méthode LoRA
    7. Entraînement via SFTTrainer (TRL)
    8. Suivi des métriques (TensorBoard) et checkpointing
    9. Évaluation finale
    10. Sauvegarde des poids LoRA + fusion optionnelle + sauvegarde du tokenizer
"""

import os
import sys
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("comptaai.train")


# ---------------------------------------------------------------------------
# Arguments CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning LoRA du modèle Qwen 3 pour ComptaAI."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen_lora.yaml",
        help="Chemin vers le fichier de configuration YAML.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Chemin d'un checkpoint à partir duquel reprendre l'entraînement.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Charge le fichier de configuration YAML centralisant tous les
    paramètres (modèle, données, LoRA, hyperparamètres, répertoires)."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration chargée depuis {config_path}")
    return config


# ---------------------------------------------------------------------------
# Environnement / reproductibilité / matériel
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fixe la graine aléatoire pour rendre les expériences reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.info(f"Seed fixée à {seed}")


def detect_device() -> str:
    """Détecte automatiquement le matériel disponible (CUDA, MPS ou CPU)
    et affiche les informations relatives au GPU utilisé."""
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU détecté : {gpu_name} ({total_mem:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple MPS détecté.")
    else:
        device = "cpu"
        logger.warning("Aucun GPU détecté, entraînement sur CPU (très lent).")

    return device


# ---------------------------------------------------------------------------
# Répertoires du projet
# ---------------------------------------------------------------------------

def setup_directories(config: dict) -> dict:
    """Crée automatiquement tous les répertoires nécessaires au projet
    (checkpoints, logs TensorBoard, poids LoRA, modèle fusionné, sorties)
    afin que le projet puisse être exécuté sans préparation manuelle."""
    dirs = config.get("directories", {})

    default_dirs = {
        "checkpoint_dir": dirs.get("checkpoint_dir", "outputs/checkpoints"),
        "logs_dir": dirs.get("logs_dir", "outputs/logs"),
        "lora_dir": dirs.get("lora_dir", "outputs/lora_weights"),
        "merged_dir": dirs.get("merged_dir", "outputs/merged_model"),
        "output_dir": dirs.get("output_dir", "outputs/final"),
    }

    for name, path in default_dirs.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Répertoire prêt [{name}] -> {path}")

    return default_dirs


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

def load_datasets(config: dict, tokenizer):
    """Charge les jeux de données d'entraînement, de validation et
    éventuellement de test au format JSONL, vérifie leurs champs
    obligatoires puis les convertit en prompts complets (instruction /
    réponse) à l'aide des templates définis dans prompt.py."""
    from dataset import load_jsonl_dataset, build_prompt_dataset

    data_cfg = config["data"]

    logger.info(f"Chargement du jeu d'entraînement : {data_cfg['train_path']}")
    train_raw = load_jsonl_dataset(data_cfg["train_path"])

    logger.info(f"Chargement du jeu de validation : {data_cfg['val_path']}")
    val_raw = load_jsonl_dataset(data_cfg["val_path"])

    test_raw = None
    if data_cfg.get("test_path"):
        logger.info(f"Chargement du jeu de test : {data_cfg['test_path']}")
        test_raw = load_jsonl_dataset(data_cfg["test_path"])

    train_dataset = build_prompt_dataset(train_raw, tokenizer, config)
    val_dataset = build_prompt_dataset(val_raw, tokenizer, config)
    test_dataset = build_prompt_dataset(test_raw, tokenizer, config) if test_raw else None

    logger.info(
        f"Jeux de données prêts : train={len(train_dataset)}, "
        f"val={len(val_dataset)}, "
        f"test={len(test_dataset) if test_dataset else 0}"
    )

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# Modèle & tokenizer (Unsloth)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(config: dict):
    """Charge le tokenizer Qwen ainsi que le modèle de base Qwen 3 via
    Unsloth, généralement en quantification 4 bits afin de réduire la
    consommation mémoire (utile sur des GPU limités, ex. Tesla T4)."""
    from unsloth import FastLanguageModel

    model_cfg = config["model"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        dtype=model_cfg.get("dtype", None),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )

    logger.info(f"Modèle de base chargé : {model_cfg['name']}")
    return model, tokenizer


def apply_lora(model, config: dict):
    """Applique la méthode LoRA : fige les poids du modèle d'origine et
    n'entraîne que de petites matrices insérées dans certaines couches
    (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).
    Affiche également le pourcentage de paramètres réellement entraînés."""
    from unsloth import FastLanguageModel

    lora_cfg = config["lora"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 16),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        random_state=config.get("training", {}).get("seed", 42),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable_params / total_params if total_params else 0

    logger.info(f"Paramètres totaux       : {total_params:,}")
    logger.info(f"Paramètres entraînables : {trainable_params:,} ({pct:.4f}%)")

    return model


# ---------------------------------------------------------------------------
# Entraînement (TRL SFTTrainer)
# ---------------------------------------------------------------------------

def build_trainer(model, tokenizer, train_dataset, val_dataset, config: dict,
                   dirs: dict):
    """Prépare le SFTTrainer de TRL en associant le modèle, le tokenizer,
    les jeux de données et l'ensemble des hyperparamètres définis dans le
    fichier de configuration. Le Trainer gère ensuite la boucle complète
    d'apprentissage (forward, backward, mise à jour LoRA, accumulation de
    gradients, précision mixte, etc.)."""
    from trl import SFTTrainer
    from transformers import TrainingArguments

    train_cfg = config["training"]

    training_args = TrainingArguments(
        output_dir=dirs["checkpoint_dir"],
        logging_dir=dirs["logs_dir"],
        num_train_epochs=train_cfg.get("num_epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        optim=train_cfg.get("optimizer", "adamw_8bit"),
        fp16=train_cfg.get("fp16", not train_cfg.get("bf16", False)),
        bf16=train_cfg.get("bf16", False),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 100),
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        seed=train_cfg.get("seed", 42),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field=train_cfg.get("dataset_text_field", "text"),
        max_seq_length=config["model"].get("max_seq_length", 2048),
        args=training_args,
        packing=train_cfg.get("packing", False),
    )

    logger.info("SFTTrainer initialisé.")
    return trainer


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_outputs(model, tokenizer, dirs: dict, config: dict) -> None:
    """Sauvegarde les poids LoRA entraînés puis, si demandé dans la
    configuration, fusionne ces poids avec le modèle de base pour produire
    un modèle autonome. Le tokenizer est sauvegardé dans les deux cas afin
    de garantir la compatibilité lors de l'inférence."""
    lora_dir = dirs["lora_dir"]
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    logger.info(f"Poids LoRA sauvegardés dans : {lora_dir}")

    if config.get("training", {}).get("merge_after_training", False):
        merged_dir = dirs["merged_dir"]
        logger.info("Fusion des poids LoRA avec le modèle de base...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Modèle fusionné sauvegardé dans : {merged_dir}")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Configuration
    config = load_config(args.config)

    # 2. Environnement
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    device = detect_device()
    logger.info(f"Device utilisé : {device}")

    # 3. Répertoires
    dirs = setup_directories(config)

    # 4. Modèle & tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # 5. Données (nécessitent le tokenizer pour la mise en forme des prompts)
    train_dataset, val_dataset, test_dataset = load_datasets(config, tokenizer)

    # 6. LoRA
    model = apply_lora(model, config)

    # 7. Trainer
    trainer = build_trainer(model, tokenizer, train_dataset, val_dataset, config, dirs)

    # 8. Entraînement (reprise depuis un checkpoint si fourni)
    logger.info("Démarrage de l'entraînement...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info("Entraînement terminé.")

    # 9. Évaluation finale
    if test_dataset is not None:
        logger.info("Évaluation sur le jeu de test...")
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        logger.info(f"Résultats de l'évaluation : {metrics}")
    else:
        logger.info("Évaluation finale sur le jeu de validation...")
        metrics = trainer.evaluate()
        logger.info(f"Résultats de l'évaluation : {metrics}")

    # 10. Sauvegarde finale
    save_outputs(model, tokenizer, dirs, config)

    logger.info("Pipeline ComptaAI terminé avec succès.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception(f"Erreur durant l'entraînement : {exc}")
        sys.exit(1)
