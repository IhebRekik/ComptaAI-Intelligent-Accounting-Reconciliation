#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py
========

Fonctions utilitaires génériques partagées par les modules du projet
ComptaAI (`evaluation.py`, `inference.py`, scripts annexes, notebooks).

Regroupe :
    - Chargement de la configuration YAML
    - Création automatique des répertoires du projet
    - Configuration du système de journalisation (logging)
    - Initialisation des graines aléatoires (seed)
    - Affichage des informations GPU
    - Sauvegarde de fichiers JSON / CSV
    - Génération de timestamps
"""

import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Charge et retourne le contenu d'un fichier de configuration YAML.

    Args:
        config_path: Chemin vers le fichier YAML (ex. configs/qwen_lora.yaml).

    Returns:
        Un dictionnaire représentant la configuration chargée.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ---------------------------------------------------------------------------
# Répertoires
# ---------------------------------------------------------------------------

def create_directories(directories: Dict[str, str]) -> None:
    """Crée automatiquement l'ensemble des répertoires listés dans le
    dictionnaire `directories` (typiquement issu de la section
    `directories` du fichier qwen_lora.yaml), s'ils n'existent pas déjà.
    """
    for name, path in directories.items():
        Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Journalisation
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "comptaai",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure et retourne un logger standardisé pour le projet.

    Ajoute un handler console et, si `log_dir` est fourni, un handler
    fichier écrivant dans `<log_dir>/<name>.log`.

    Args:
        name: Nom du logger.
        log_dir: Répertoire où écrire le fichier de log (optionnel).
        level: Niveau de journalisation.

    Returns:
        Le logger configuré.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # déjà configuré

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fixe la graine aléatoire pour Python, NumPy et PyTorch (si
    disponible), afin de garantir la reproductibilité des expériences."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Informations matérielles
# ---------------------------------------------------------------------------

def display_gpu_info() -> Dict[str, Any]:
    """Détecte et affiche les informations sur le GPU disponible
    (nom, mémoire totale) ou signale l'absence de GPU / usage du CPU.

    Returns:
        Un dictionnaire décrivant le device détecté.
    """
    try:
        import torch
    except ImportError:
        return {"device": "cpu", "reason": "torch non installé"}

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        info = {"device": "cuda", "gpu_name": name, "total_memory_gb": round(total_mem_gb, 2)}
    elif torch.backends.mps.is_available():
        info = {"device": "mps"}
    else:
        info = {"device": "cpu"}

    print(f"[GPU INFO] {info}")
    return info


# ---------------------------------------------------------------------------
# Sauvegarde de fichiers
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Sauvegarde un objet Python au format JSON, en créant le répertoire
    parent si nécessaire."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Sauvegarde une liste de dictionnaires au format CSV. Les clés du
    premier dictionnaire déterminent les colonnes du fichier."""
    if not rows:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def generate_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Génère un timestamp lisible, utile pour nommer des fichiers de
    sortie (checkpoints, exports, résultats d'évaluation) de manière
    unique."""
    return time.strftime(fmt)
