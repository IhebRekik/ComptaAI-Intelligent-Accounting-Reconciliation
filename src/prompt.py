#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prompt.py
=========

Module responsable de la construction des prompts utilisés pour
l'entraînement et l'inférence du modèle Qwen 3 dans le projet ComptaAI.

Ce module expose une fonction unique, `build_training_prompt`, utilisée
par `dataset.py` pour transformer chaque exemple brut (instruction /
réponse) en un texte complet prêt à être tokenisé, au format ChatML
attendu par les modèles de la famille Qwen.
"""

SYSTEM_PROMPT = (
    "Tu es ComptaAI, un assistant expert en comptabilité. "
    "Tu réponds de manière précise, claire et professionnelle aux questions "
    "comptables, tu expliques les écritures comptables, tu assistes au "
    "lettrage comptable et tu analyses les opérations financières."
)


def build_training_prompt(instruction: str, response: str = "") -> str:
    """Construit un prompt complet au format ChatML utilisé par Qwen.

    Cette fonction est le point d'entrée unique pour la construction des
    prompts dans le projet. Elle est utilisée à deux fins :

    - Pendant l'entraînement : `response` est fourni, et le texte retourné
      contient l'échange complet (system + user + assistant), utilisé tel
      quel comme cible de tokenisation par `dataset.py`.
    - Pendant l'inférence : `response` est laissé vide (chaîne vide par
      défaut), et le texte retourné se termine juste après le tag
      `<|im_start|>assistant`, prêt à être complété par le modèle via
      `inference.py`.

    Args:
        instruction: La question ou l'instruction de l'utilisateur
            (ex. "Comment lettrer une facture partiellement payée ?").
        response: La réponse attendue du modèle. Laisser vide pour
            générer un prompt d'inférence (sans réponse).

    Returns:
        Le prompt complet, formaté en ChatML, sous forme de chaîne de
        caractères.

    Raises:
        ValueError: si `instruction` est vide ou n'est pas une chaîne.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("`instruction` doit être une chaîne de caractères non vide.")

    instruction = instruction.strip()
    response = response.strip() if isinstance(response, str) else ""

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if response:
        prompt += f"{response}<|im_end|>"

    return prompt
