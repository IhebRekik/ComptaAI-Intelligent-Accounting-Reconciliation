#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
webui.py
========

Interface Web de supervision du fine-tuning ComptaAI, construite avec
**Gradio** et exposée sur Internet via **ngrok**. Cette interface ne
dialogue PAS avec le modèle entraîné (ce n'est pas un chatbot) : son
unique rôle est de piloter et superviser le processus de fine-tuning
LoRA de Qwen3-8B, en s'appuyant sur les modules existants du projet
(`train.py`, `dataset.py`, `utils.py`, `configs/qwen_lora.yaml`), sans
jamais les modifier.

Sections du tableau de bord :
    1. Système       — GPU, CUDA, Python, PyTorch, Google Drive
    2. Configuration  — chargement et vérification de qwen_lora.yaml
    3. Modèle         — vérification / téléchargement du modèle de base
    4. Dataset        — statistiques, validité, aperçu des exemples
    5. Entraînement   — démarrage / reprise / arrêt + suivi temps réel + console
    6. Checkpoints    — liste, reprise, sauvegarde manuelle, suppression

Lancement :
    python src/webui.py
"""

from __future__ import annotations

import os
import re
import sys
import shutil
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# ---------------------------------------------------------------------------
# Chemins du projet & imports des modules ComptaAI existants
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils import load_yaml_config, generate_timestamp  # noqa: E402
from dataset import load_jsonl_dataset, dataset_statistics  # noqa: E402

DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "configs" / "qwen_lora.yaml")
TRAIN_SCRIPT_PATH = str(SRC_DIR / "train.py")


def _resolve(path_str: str) -> Path:
    """Résout un chemin relatif du fichier de configuration par rapport
    à la racine du projet, afin que l'interface fonctionne quel que soit
    le répertoire courant depuis lequel `webui.py` est lancé."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# =============================================================================
# 1. SECTION SYSTÈME
# =============================================================================

def _get_python_version() -> str:
    return sys.version.split()[0]


def _get_pytorch_info() -> Dict[str, Any]:
    """Retourne la version de PyTorch et la disponibilité de CUDA."""
    try:
        import torch
        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }
    except ImportError:
        return {"installed": False}


def _get_gpu_info() -> Dict[str, Any]:
    """Détecte le GPU disponible et sa mémoire totale via PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "available": True,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            }
        return {"available": False}
    except ImportError:
        return {"available": False}


def get_gpu_memory_usage() -> Optional[Tuple[float, float]]:
    """Interroge `nvidia-smi` pour obtenir la mémoire GPU utilisée / totale
    (en Go), indépendamment du processus qui l'utilise. Retourne `None`
    si `nvidia-smi` n'est pas disponible (pas de GPU NVIDIA)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        used_mb, total_mb = map(float, result.stdout.strip().split(","))
        return round(used_mb / 1024, 2), round(total_mb / 1024, 2)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def _is_drive_mounted() -> bool:
    """Vérifie si Google Drive est monté (chemin standard sous Colab)."""
    return Path("/content/drive/MyDrive").exists()


def build_system_info_markdown(config_path: str) -> str:
    """Construit le rapport Markdown de la section « Système »."""
    config = _safe_load_config(config_path)
    project_cfg = config.get("project", {}) if config else {}
    model_cfg = config.get("model", {}) if config else {}

    torch_info = _get_pytorch_info()
    gpu_info = _get_gpu_info()
    gpu_mem = get_gpu_memory_usage()
    drive_mounted = _is_drive_mounted()

    lines = [
        "### 🖥️ Informations système\n",
        f"| Élément | Valeur |",
        f"|---|---|",
        f"| **Projet** | {project_cfg.get('name', 'ComptaAI')} (v{project_cfg.get('version', '—')}) |",
        f"| **Modèle configuré** | {model_cfg.get('name', '—')} |",
        f"| **Python** | {_get_python_version()} |",
    ]

    if torch_info.get("installed"):
        lines.append(f"| **PyTorch** | {torch_info['version']} |")
        cuda_status = "✅ Disponible" if torch_info["cuda_available"] else "❌ Indisponible"
        lines.append(f"| **CUDA** | {cuda_status} (torch build: {torch_info.get('cuda_version') or '—'}) |")
    else:
        lines.append("| **PyTorch** | ⚠️ Non installé |")

    if gpu_info.get("available"):
        lines.append(f"| **GPU détecté** | {gpu_info['name']} |")
        lines.append(f"| **VRAM totale** | {gpu_info['total_memory_gb']} Go |")
    else:
        lines.append("| **GPU détecté** | ❌ Aucun GPU CUDA détecté |")

    if gpu_mem is not None:
        used, total = gpu_mem
        pct = round(100 * used / total, 1) if total else 0
        lines.append(f"| **Mémoire GPU utilisée** | {used} / {total} Go ({pct}%) |")

    drive_status = "✅ Monté (/content/drive/MyDrive)" if drive_mounted else "❌ Non monté"
    lines.append(f"| **Google Drive** | {drive_status} |")

    return "\n".join(lines)


# =============================================================================
# 2. SECTION CONFIGURATION
# =============================================================================

def _safe_load_config(config_path: str) -> Optional[Dict[str, Any]]:
    try:
        return load_yaml_config(config_path)
    except Exception:
        return None


def verify_configuration(config_path: str) -> str:
    """Charge `qwen_lora.yaml` et vérifie que tous les chemins de données
    et répertoires qu'il référence sont valides. Retourne un rapport
    Markdown détaillé, utilisé par la section « Configuration »."""
    report = [f"### ⚙️ Vérification de la configuration\n`{config_path}`\n"]

    try:
        config = load_yaml_config(config_path)
    except FileNotFoundError:
        return "\n".join(report) + "\n❌ **Fichier de configuration introuvable.**"
    except Exception as exc:
        return "\n".join(report) + f"\n❌ **Erreur de lecture du YAML :** {exc}"

    report.append("✅ Fichier de configuration chargé avec succès.\n")

    # --- Jeux de données ---
    report.append("**Jeux de données :**\n")
    data_cfg = config.get("data", {})
    for key, label in [("train_path", "Train"), ("val_path", "Validation"), ("test_path", "Test")]:
        path_str = data_cfg.get(key)
        if not path_str:
            report.append(f"- {label} : ⚠️ chemin non défini dans la configuration")
            continue
        full_path = _resolve(path_str)
        if full_path.exists():
            n_lines = sum(1 for line in open(full_path, encoding="utf-8") if line.strip())
            report.append(f"- {label} (`{path_str}`) : ✅ trouvé, {n_lines} lignes")
        else:
            report.append(f"- {label} (`{path_str}`) : ❌ fichier introuvable")

    # --- Répertoires de sauvegarde ---
    report.append("\n**Répertoires de sauvegarde :**\n")
    dirs_cfg = config.get("directories", {})
    for key, label in [
        ("checkpoint_dir", "Checkpoints"), ("logs_dir", "Logs"),
        ("lora_dir", "Adaptateurs LoRA"), ("merged_dir", "Modèle fusionné"),
        ("output_dir", "Exports"),
    ]:
        path_str = dirs_cfg.get(key)
        if not path_str:
            report.append(f"- {label} : ⚠️ non défini")
            continue
        full_path = _resolve(path_str)
        exists = full_path.exists()
        status = "✅ existe déjà" if exists else "ℹ️ sera créé automatiquement par train.py"
        report.append(f"- {label} (`{path_str}`) : {status}")

    # --- Sections obligatoires ---
    report.append("\n**Sections de configuration :**\n")
    for section in ["project", "model", "data", "directories", "lora", "training", "generation"]:
        status = "✅" if section in config else "❌ manquante"
        report.append(f"- `{section}` : {status}")

    return "\n".join(report)


# =============================================================================
# 3. SECTION MODÈLE (téléchargement Hugging Face / Unsloth)
# =============================================================================

def is_model_cached(model_name: str) -> bool:
    """Vérifie si le modèle est déjà présent dans le cache local
    Hugging Face (`~/.cache/huggingface/hub`)."""
    try:
        from huggingface_hub import scan_cache_info
        cache_info = scan_cache_info()
        cached_repos = {repo.repo_id for repo in cache_info.repos}
        return model_name in cached_repos
    except Exception:
        return False


def check_model_status(config_path: str) -> str:
    config = _safe_load_config(config_path)
    if not config:
        return "❌ Impossible de charger la configuration."

    model_name = config.get("model", {}).get("name", "")
    if not model_name:
        return "⚠️ Aucun modèle défini dans la configuration (`model.name`)."

    cached = is_model_cached(model_name)
    status = "✅ Présent dans le cache local" if cached else "❌ Non présent — téléchargement nécessaire"
    return f"### 📦 Modèle configuré : `{model_name}`\n\n**Statut du cache :** {status}"


def download_model(config_path: str, progress: gr.Progress = gr.Progress(track_tqdm=True)) -> str:
    """Télécharge le modèle de base configuré depuis Hugging Face si celui-ci
    n'est pas déjà présent dans le cache local. `gr.Progress(track_tqdm=True)`
    intercepte automatiquement la barre de progression `tqdm` utilisée en
    interne par `huggingface_hub` pendant le téléchargement des poids."""
    config = _safe_load_config(config_path)
    if not config:
        return "❌ Impossible de charger la configuration."

    model_name = config.get("model", {}).get("name", "")
    if not model_name:
        return "⚠️ Aucun modèle défini dans la configuration (`model.name`)."

    if is_model_cached(model_name):
        return f"✅ Le modèle `{model_name}` est déjà présent dans le cache local. Aucun téléchargement nécessaire."

    try:
        from huggingface_hub import snapshot_download
        progress(0, desc=f"Téléchargement de {model_name}...")
        snapshot_download(repo_id=model_name)
        progress(1.0, desc="Terminé")
        return f"✅ Modèle `{model_name}` téléchargé avec succès dans le cache local."
    except Exception as exc:
        return f"❌ Échec du téléchargement : {exc}"


# =============================================================================
# 4. SECTION DATASET
# =============================================================================

def analyze_datasets(config_path: str) -> Tuple[str, str]:
    """Charge et valide les trois jeux de données via `dataset.py`,
    retourne un rapport de statistiques et un aperçu d'exemples."""
    config = _safe_load_config(config_path)
    if not config:
        return "❌ Impossible de charger la configuration.", ""

    data_cfg = config.get("data", {})
    min_len = data_cfg.get("min_response_length", 1)

    stats_lines = ["### 📊 Statistiques des jeux de données\n"]
    preview_lines = ["### 👀 Aperçu des exemples\n"]

    for key, label in [("train_path", "Train"), ("val_path", "Validation"), ("test_path", "Test")]:
        path_str = data_cfg.get(key)
        if not path_str:
            stats_lines.append(f"**{label}** : ⚠️ chemin non défini\n")
            continue

        full_path = _resolve(path_str)
        try:
            examples = load_jsonl_dataset(str(full_path), min_response_length=min_len)
        except FileNotFoundError:
            stats_lines.append(f"**{label}** (`{path_str}`) : ❌ fichier introuvable\n")
            continue
        except Exception as exc:
            stats_lines.append(f"**{label}** (`{path_str}`) : ❌ erreur — {exc}\n")
            continue

        stats = dataset_statistics(examples)
        stats_lines.append(f"**{label}** (`{path_str}`) : ✅ {stats.get('count', 0)} exemples valides")
        if stats.get("count", 0) > 0:
            stats_lines.append(
                f"  - Longueur instruction (car.) : min={stats['instruction_length']['min']}, "
                f"max={stats['instruction_length']['max']}, "
                f"moy={stats['instruction_length']['avg']:.0f}"
            )
            stats_lines.append(
                f"  - Longueur réponse (car.) : min={stats['response_length']['min']}, "
                f"max={stats['response_length']['max']}, "
                f"moy={stats['response_length']['avg']:.0f}\n"
            )

        if examples:
            preview_lines.append(f"**{label} — premier exemple :**")
            preview_lines.append(f"- Instruction : {examples[0]['instruction']}")
            preview_lines.append(f"- Réponse : {examples[0]['response'][:300]}"
                                  + ("..." if len(examples[0]["response"]) > 300 else ""))
            preview_lines.append("")

    return "\n".join(stats_lines), "\n".join(preview_lines)


# =============================================================================
# 5. SECTION ENTRAÎNEMENT — gestionnaire de processus
# =============================================================================

# Expressions régulières utilisées pour extraire les métriques des logs
# produits par `train.py` (Trainer / tqdm). Le format exact peut varier
# légèrement selon les versions de `transformers` / `trl` ; ajuster ces
# patterns si nécessaire.
_RE_TQDM_PROGRESS = re.compile(
    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[(\d+):(\d+)<(?:(\d+):(\d+)|\?),\s*([\d.]+)(it/s|s/it)\]"
)
_RE_LOSS = re.compile(r"'loss':\s*([\d.eE+-]+)")
_RE_EVAL_LOSS = re.compile(r"'eval_loss':\s*([\d.eE+-]+)")
_RE_LR = re.compile(r"'learning_rate':\s*([\d.eE+-]+)")
_RE_EPOCH = re.compile(r"'epoch':\s*([\d.]+)")
_RE_TOTAL_EPOCHS = re.compile(r"num_epochs['\"]?\s*[:=]\s*(\d+)")

MAX_LOG_LINES = 2000


def _default_metrics() -> Dict[str, Any]:
    return {
        "epoch": None,
        "step": None,
        "total_steps": None,
        "progress_pct": 0,
        "loss": None,
        "eval_loss": None,
        "learning_rate": None,
        "elapsed": None,
        "eta": None,
        "speed": None,
    }


class TrainingManager:
    """Gère le cycle de vie du sous-processus `train.py` : démarrage,
    lecture des logs en temps réel dans un thread dédié, extraction des
    métriques d'entraînement, et arrêt propre.

    `train.py` n'est jamais importé ni modifié : il est exécuté comme un
    script indépendant (`python train.py --config ...`), exactement comme
    depuis un terminal, ce qui garantit une compatibilité totale avec son
    fonctionnement existant.
    """

    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.log_lines: List[str] = []
        self.metrics: Dict[str, Any] = _default_metrics()
        self.status: str = "idle"  # idle | running | completed | failed | stopped
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, config_path: str, resume_checkpoint: Optional[str] = None) -> str:
        if self.is_running():
            return "⚠️ Un entraînement est déjà en cours."

        if not Path(config_path).exists() and not _resolve(config_path).exists():
            return f"❌ Fichier de configuration introuvable : {config_path}"

        cmd = [sys.executable, "-u", TRAIN_SCRIPT_PATH, "--config", config_path]
        if resume_checkpoint:
            cmd += ["--resume_from_checkpoint", resume_checkpoint]

        with self._lock:
            self.log_lines = []
            self.metrics = _default_metrics()
            self.status = "running"
            self.start_time = time.time()
            self.end_time = None

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            with self._lock:
                self.status = "failed"
            return f"❌ Impossible de démarrer l'entraînement : {exc}"

        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

        resume_note = f" (reprise depuis `{resume_checkpoint}`)" if resume_checkpoint else ""
        return f"🚀 Entraînement démarré{resume_note}."

    def _reader_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for raw_line in self.process.stdout:
            line = raw_line.rstrip("\n")
            with self._lock:
                self.log_lines.append(line)
                if len(self.log_lines) > MAX_LOG_LINES:
                    self.log_lines = self.log_lines[-MAX_LOG_LINES:]
                self._parse_line(line)

        return_code = self.process.wait()
        with self._lock:
            self.end_time = time.time()
            if self.status == "stopped":
                pass  # déjà marqué comme arrêté manuellement
            elif return_code == 0:
                self.status = "completed"
            else:
                self.status = "failed"

    def _parse_line(self, line: str) -> None:
        """Met à jour `self.metrics` à partir d'une ligne de log. Doit être
        appelée sous `self._lock` (déjà le cas dans `_reader_loop`)."""
        m = _RE_TQDM_PROGRESS.search(line)
        if m:
            pct, step, total, mm, ss, eta_mm, eta_ss, speed, unit = m.groups()
            self.metrics["progress_pct"] = int(pct)
            self.metrics["step"] = int(step)
            self.metrics["total_steps"] = int(total)
            self.metrics["elapsed"] = f"{mm}:{ss}"
            self.metrics["eta"] = f"{eta_mm}:{eta_ss}" if eta_mm else "—"
            self.metrics["speed"] = f"{speed} {unit}"

        loss_match = _RE_LOSS.search(line)
        if loss_match:
            self.metrics["loss"] = float(loss_match.group(1))

        eval_loss_match = _RE_EVAL_LOSS.search(line)
        if eval_loss_match:
            self.metrics["eval_loss"] = float(eval_loss_match.group(1))

        lr_match = _RE_LR.search(line)
        if lr_match:
            self.metrics["learning_rate"] = float(lr_match.group(1))

        epoch_match = _RE_EPOCH.search(line)
        if epoch_match:
            self.metrics["epoch"] = float(epoch_match.group(1))

    def stop(self) -> str:
        if not self.is_running():
            return "ℹ️ Aucun entraînement en cours."

        with self._lock:
            self.status = "stopped"

        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()

        with self._lock:
            self.end_time = time.time()

        return "⏹️ Entraînement arrêté."

    def snapshot(self) -> Dict[str, Any]:
        """Retourne une copie cohérente de l'état courant (logs + métriques)
        pour affichage dans l'interface."""
        with self._lock:
            return {
                "status": self.status,
                "metrics": dict(self.metrics),
                "logs": "\n".join(self.log_lines[-500:]),
                "start_time": self.start_time,
                "end_time": self.end_time,
            }


# Instance unique partagée par toute l'interface (un seul entraînement à la fois)
_training_manager = TrainingManager()


def start_training_cb(config_path: str, resume_checkpoint: Optional[str]) -> str:
    checkpoint = resume_checkpoint if resume_checkpoint and resume_checkpoint != "(aucun)" else None
    return _training_manager.start(config_path, checkpoint)


def stop_training_cb() -> str:
    return _training_manager.stop()


def _format_elapsed(start_time: Optional[float], end_time: Optional[float]) -> str:
    if not start_time:
        return "—"
    end = end_time or time.time()
    return str(timedelta(seconds=int(end - start_time)))


def refresh_dashboard(config_path: str) -> Tuple[str, int, str, str, str]:
    """Fonction de rafraîchissement périodique du tableau de bord
    d'entraînement, appelée par le `gr.Timer` de l'onglet Entraînement.

    Returns:
        (status_markdown, progress_percent, metrics_markdown, console_text, results_markdown)
    """
    snap = _training_manager.snapshot()
    status = snap["status"]
    metrics = snap["metrics"]

    status_labels = {
        "idle": "⚪ Inactif",
        "running": "🟢 Entraînement en cours...",
        "completed": "✅ Entraînement terminé avec succès",
        "failed": "❌ Entraînement en échec (voir la console)",
        "stopped": "⏹️ Entraînement arrêté manuellement",
    }
    status_md = f"**Statut : {status_labels.get(status, status)}**  \nTemps écoulé : {_format_elapsed(snap['start_time'], snap['end_time'])}"

    progress_pct = metrics.get("progress_pct") or 0

    gpu_mem = get_gpu_memory_usage()
    gpu_mem_str = f"{gpu_mem[0]} / {gpu_mem[1]} Go" if gpu_mem else "—"

    metrics_md_lines = [
        "### 📈 Suivi en temps réel\n",
        f"| Métrique | Valeur |",
        f"|---|---|",
        f"| Époque | {metrics.get('epoch') if metrics.get('epoch') is not None else '—'} |",
        f"| Itération | {metrics.get('step') or '—'} / {metrics.get('total_steps') or '—'} |",
        f"| Progression | {progress_pct}% |",
        f"| Loss (entraînement) | {metrics.get('loss') if metrics.get('loss') is not None else '—'} |",
        f"| Loss (évaluation) | {metrics.get('eval_loss') if metrics.get('eval_loss') is not None else '—'} |",
        f"| Taux d'apprentissage | {metrics.get('learning_rate') if metrics.get('learning_rate') is not None else '—'} |",
        f"| Vitesse | {metrics.get('speed') or '—'} |",
        f"| Temps restant estimé | {metrics.get('eta') or '—'} |",
        f"| Mémoire GPU | {gpu_mem_str} |",
    ]
    metrics_md = "\n".join(metrics_md_lines)

    console_text = snap["logs"] or "(aucun log pour le moment)"

    results_md = ""
    if status in ("completed", "failed", "stopped"):
        results_md = build_results_summary(config_path, snap)

    return status_md, progress_pct, metrics_md, console_text, results_md


def build_results_summary(config_path: str, snap: Dict[str, Any]) -> str:
    """Construit le résumé final affiché une fois l'entraînement terminé,
    arrêté ou en échec : durée, dernière perte connue, fichiers produits."""
    config = _safe_load_config(config_path) or {}
    dirs_cfg = config.get("directories", {})
    metrics = snap["metrics"]

    duration = _format_elapsed(snap["start_time"], snap["end_time"])

    lines = ["### 🏁 Résumé de l'entraînement\n"]
    lines.append(f"- **Durée totale** : {duration}")
    lines.append(f"- **Dernière perte (train)** : {metrics.get('loss') if metrics.get('loss') is not None else '—'}")
    lines.append(f"- **Dernière perte (eval)** : {metrics.get('eval_loss') if metrics.get('eval_loss') is not None else '—'}")
    lines.append(f"- **Époques atteintes** : {metrics.get('epoch') if metrics.get('epoch') is not None else '—'}")

    for key, label in [("lora_dir", "Adaptateurs LoRA"), ("merged_dir", "Modèle fusionné"),
                        ("checkpoint_dir", "Checkpoints")]:
        path_str = dirs_cfg.get(key)
        if not path_str:
            continue
        full_path = _resolve(path_str)
        if full_path.exists():
            n_files = sum(1 for _ in full_path.rglob("*") if _.is_file())
            lines.append(f"- **{label}** : `{path_str}` ({n_files} fichiers)")
        else:
            lines.append(f"- **{label}** : `{path_str}` (non généré)")

    return "\n".join(lines)


# =============================================================================
# 6. SECTION CHECKPOINTS
# =============================================================================

def _checkpoint_dir_from_config(config_path: str) -> Optional[Path]:
    config = _safe_load_config(config_path)
    if not config:
        return None
    path_str = config.get("directories", {}).get("checkpoint_dir")
    return _resolve(path_str) if path_str else None


def list_checkpoints(config_path: str) -> List[str]:
    """Retourne la liste des checkpoints disponibles, triés du plus
    récent au plus ancien."""
    ckpt_dir = _checkpoint_dir_from_config(config_path)
    if not ckpt_dir or not ckpt_dir.exists():
        return []

    checkpoints = [
        p for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    ]
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in checkpoints]


def refresh_checkpoints_cb(config_path: str) -> Tuple[str, "gr.update", "gr.update"]:
    checkpoints = list_checkpoints(config_path)

    if not checkpoints:
        listing_md = "Aucun checkpoint disponible pour le moment."
    else:
        listing_md = "### 💾 Checkpoints disponibles\n\n" + "\n".join(
            f"- `{Path(c).name}`" for c in checkpoints
        )

    resume_choices = ["(aucun)"] + checkpoints
    delete_choices = checkpoints

    return (
        listing_md,
        gr.update(choices=resume_choices, value="(aucun)"),
        gr.update(choices=delete_choices, value=None),
    )


def delete_checkpoint_cb(config_path: str, checkpoint_path: Optional[str]) -> str:
    if not checkpoint_path:
        return "⚠️ Sélectionnez un checkpoint à supprimer."

    path = Path(checkpoint_path)
    if not path.exists():
        return f"❌ Checkpoint introuvable : {checkpoint_path}"

    try:
        shutil.rmtree(path)
        return f"🗑️ Checkpoint supprimé : `{path.name}`"
    except Exception as exc:
        return f"❌ Erreur lors de la suppression : {exc}"


def manual_save_checkpoint_cb(config_path: str) -> str:
    """Effectue une sauvegarde manuelle « de sécurité » en dupliquant le
    checkpoint le plus récent dans un dossier horodaté séparé
    (`checkpoints/manual_saves/`). Cette fonction ne peut pas forcer
    `train.py` à sauvegarder immédiatement en cours d'entraînement (le
    Trainer sauvegarde selon `save_steps` défini dans la configuration) ;
    elle protège en revanche le dernier checkpoint existant contre une
    éventuelle suppression ou rotation (`save_total_limit`)."""
    ckpt_dir = _checkpoint_dir_from_config(config_path)
    if not ckpt_dir or not ckpt_dir.exists():
        return "❌ Aucun répertoire de checkpoints trouvé."

    checkpoints = list_checkpoints(config_path)
    if not checkpoints:
        return "⚠️ Aucun checkpoint existant à sauvegarder pour le moment."

    latest = Path(checkpoints[0])
    manual_dir = ckpt_dir / "manual_saves" / f"{latest.name}_{generate_timestamp()}"

    try:
        shutil.copytree(latest, manual_dir)
        return f"✅ Sauvegarde manuelle créée : `{manual_dir.relative_to(PROJECT_ROOT)}`"
    except Exception as exc:
        return f"❌ Échec de la sauvegarde manuelle : {exc}"


# =============================================================================
# CONSTRUCTION DE L'INTERFACE GRADIO
# =============================================================================

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="ComptaAI — Supervision du fine-tuning") as demo:
        gr.Markdown(
            "# 🧮 ComptaAI — Tableau de bord de fine-tuning\n"
            "Supervision du fine-tuning LoRA de Qwen3-8B. "
            "Cette interface ne dialogue pas avec le modèle : elle pilote uniquement l'entraînement."
        )

        config_path_box = gr.Textbox(
            value=DEFAULT_CONFIG_PATH,
            label="📄 Fichier de configuration (qwen_lora.yaml)",
        )

        with gr.Tabs():

            # ---------------- Onglet Système ----------------
            with gr.Tab("🖥️ Système"):
                sysinfo_md = gr.Markdown()
                refresh_sys_btn = gr.Button("Actualiser")
                refresh_sys_btn.click(fn=build_system_info_markdown, inputs=config_path_box, outputs=sysinfo_md)
                demo.load(fn=build_system_info_markdown, inputs=config_path_box, outputs=sysinfo_md)

            # ---------------- Onglet Configuration ----------------
            with gr.Tab("⚙️ Configuration"):
                verify_btn = gr.Button("Vérifier la configuration", variant="primary")
                config_report_md = gr.Markdown()
                verify_btn.click(fn=verify_configuration, inputs=config_path_box, outputs=config_report_md)

            # ---------------- Onglet Modèle ----------------
            with gr.Tab("📦 Modèle"):
                model_status_md = gr.Markdown()
                with gr.Row():
                    check_model_btn = gr.Button("Vérifier le cache")
                    download_model_btn = gr.Button("Télécharger le modèle", variant="primary")
                check_model_btn.click(fn=check_model_status, inputs=config_path_box, outputs=model_status_md)
                download_model_btn.click(fn=download_model, inputs=config_path_box, outputs=model_status_md)

            # ---------------- Onglet Dataset ----------------
            with gr.Tab("📊 Dataset"):
                analyze_btn = gr.Button("Analyser les jeux de données", variant="primary")
                with gr.Row():
                    dataset_stats_md = gr.Markdown()
                    dataset_preview_md = gr.Markdown()
                analyze_btn.click(fn=analyze_datasets, inputs=config_path_box,
                                   outputs=[dataset_stats_md, dataset_preview_md])

            # ---------------- Onglet Entraînement ----------------
            with gr.Tab("🚀 Entraînement"):
                with gr.Row():
                    resume_dropdown = gr.Dropdown(
                        label="Reprendre depuis un checkpoint (optionnel)",
                        choices=["(aucun)"], value="(aucun)",
                    )
                    refresh_ckpt_for_resume_btn = gr.Button("🔄 Actualiser la liste")

                with gr.Row():
                    start_btn = gr.Button("▶️ Démarrer l'entraînement", variant="primary")
                    stop_btn = gr.Button("⏹️ Arrêter l'entraînement", variant="stop")

                action_status_md = gr.Markdown()
                status_md = gr.Markdown("**Statut : ⚪ Inactif**")
                progress_bar = gr.Slider(minimum=0, maximum=100, value=0, step=1,
                                          label="Progression (%)", interactive=False)
                metrics_md = gr.Markdown()
                console = gr.Textbox(
                    label="Console — logs en direct de train.py",
                    lines=18, max_lines=18, interactive=False, autoscroll=True,
                )
                results_md = gr.Markdown()

                timer = gr.Timer(2.0, active=True)

                start_btn.click(
                    fn=start_training_cb, inputs=[config_path_box, resume_dropdown], outputs=action_status_md,
                )
                stop_btn.click(fn=stop_training_cb, outputs=action_status_md)

                timer.tick(
                    fn=refresh_dashboard,
                    inputs=config_path_box,
                    outputs=[status_md, progress_bar, metrics_md, console, results_md],
                )

            # ---------------- Onglet Checkpoints ----------------
            with gr.Tab("💾 Checkpoints"):
                refresh_ckpt_btn = gr.Button("🔄 Actualiser")
                checkpoints_listing_md = gr.Markdown()

                with gr.Row():
                    delete_dropdown = gr.Dropdown(label="Supprimer un checkpoint", choices=[])
                    delete_btn = gr.Button("🗑️ Supprimer", variant="stop")

                manual_save_btn = gr.Button("💾 Sauvegarde manuelle du dernier checkpoint")
                checkpoint_action_md = gr.Markdown()

                refresh_ckpt_btn.click(
                    fn=refresh_checkpoints_cb, inputs=config_path_box,
                    outputs=[checkpoints_listing_md, resume_dropdown, delete_dropdown],
                )
                refresh_ckpt_for_resume_btn.click(
                    fn=refresh_checkpoints_cb, inputs=config_path_box,
                    outputs=[checkpoints_listing_md, resume_dropdown, delete_dropdown],
                )
                delete_btn.click(
                    fn=delete_checkpoint_cb, inputs=[config_path_box, delete_dropdown],
                    outputs=checkpoint_action_md,
                )
                manual_save_btn.click(
                    fn=manual_save_checkpoint_cb, inputs=config_path_box, outputs=checkpoint_action_md,
                )

                demo.load(fn=refresh_checkpoints_cb, inputs=config_path_box,
                           outputs=[checkpoints_listing_md, resume_dropdown, delete_dropdown])

    return demo


# =============================================================================
# LANCEMENT DE L'APPLICATION (avec tunnel ngrok)
# =============================================================================

def _start_ngrok_tunnel(port: int, authtoken: str = "") -> Optional[str]:
    """Ouvre un tunnel ngrok vers le port local de Gradio et retourne
    l'URL publique générée, ou `None` en cas d'échec."""
    try:
        from pyngrok import ngrok
    except ImportError:
        print("[ComptaAI] pyngrok n'est pas installé — impossible de créer un tunnel ngrok.")
        print("           Installez-le avec : pip install pyngrok")
        return None

    token = authtoken or os.environ.get("NGROK_AUTHTOKEN", "")
    if token:
        ngrok.set_auth_token(token)
    else:
        print("[ComptaAI] Aucun jeton ngrok fourni (config `webui.ngrok_authtoken` ou "
              "variable d'environnement NGROK_AUTHTOKEN) — le tunnel peut être limité.")

    try:
        tunnel = ngrok.connect(port, "http")
        return tunnel.public_url
    except Exception as exc:
        print(f"[ComptaAI] Échec de la création du tunnel ngrok : {exc}")
        return None


def main() -> None:
    config = _safe_load_config(DEFAULT_CONFIG_PATH) or {}
    webui_cfg = config.get("webui", {})

    port = webui_cfg.get("server_port", 7860)
    use_ngrok = webui_cfg.get("use_ngrok", True)

    demo = build_interface()

    public_url = None
    if use_ngrok:
        public_url = _start_ngrok_tunnel(port, webui_cfg.get("ngrok_authtoken", ""))

    if public_url:
        print(f"\n[ComptaAI] 🌐 Interface accessible publiquement : {public_url}\n")
    else:
        print(f"\n[ComptaAI] Interface accessible localement sur : http://localhost:{port}\n")

    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
