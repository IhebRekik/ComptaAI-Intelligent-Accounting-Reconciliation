#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference.py
============

Module d'inférence du projet ComptaAI. Permet d'utiliser le modèle
entraîné (fusionné ou LoRA) en production pour répondre à des questions
comptables.

Utilisation en ligne de commande :
    python src/inference.py --config configs/qwen_lora.yaml --mode merged \
        --instruction "Comment lettrer une facture partiellement payée ?"

Utilisation programmatique :
    from inference import ComptaAIAssistant
    assistant = ComptaAIAssistant(config_path="configs/qwen_lora.yaml", mode="merged")
    reponse = assistant.generate("Comment comptabiliser une note de frais ?")
"""

import argparse
from typing import Any, Dict, Optional

from prompt import build_training_prompt
from utils import load_yaml_config, setup_logger

logger = setup_logger("comptaai.inference")


class ComptaAIAssistant:
    """Enveloppe le modèle ComptaAI entraîné pour une utilisation simple
    en inférence : chargement du modèle (fusionné ou LoRA), construction
    du prompt utilisateur, tokenisation, génération et décodage de la
    réponse."""

    def __init__(self, config_path: str = "configs/qwen_lora.yaml", mode: str = "merged"):
        """Initialise l'assistant.

        Args:
            config_path: Chemin vers le fichier de configuration YAML.
            mode: "merged" pour charger le modèle fusionné autonome,
                "lora" pour charger le modèle de base avec les adaptateurs
                LoRA.
        """
        self.config: Dict[str, Any] = load_yaml_config(config_path)
        self.mode = mode
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        from unsloth import FastLanguageModel

        dirs = self.config["directories"]
        model_cfg = self.config["model"]

        model_path = dirs["merged_dir"] if self.mode == "merged" else dirs["lora_dir"]
        logger.info(f"Chargement du modèle ({self.mode}) depuis : {model_path}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=model_cfg.get("max_seq_length", 2048),
            dtype=model_cfg.get("dtype", None),
            load_in_4bit=model_cfg.get("load_in_4bit", True),
        )

        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def generate(self, instruction: str, generation_overrides: Optional[Dict[str, Any]] = None) -> str:
        """Génère une réponse comptable à partir d'une instruction
        utilisateur.

        Args:
            instruction: La question posée par l'utilisateur.
            generation_overrides: Dictionnaire optionnel pour surcharger
                ponctuellement les paramètres de génération définis dans
                la configuration (section `generation`).

        Returns:
            La réponse générée par le modèle, sous forme de texte brut.
        """
        gen_cfg = dict(self.config.get("generation", {}))
        if generation_overrides:
            gen_cfg.update(generation_overrides)

        prompt = build_training_prompt(instruction, response="")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 512),
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            top_k=gen_cfg.get("top_k", 50),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.1),
            do_sample=gen_cfg.get("do_sample", True),
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inférence avec le modèle ComptaAI.")
    parser.add_argument("--config", type=str, default="configs/qwen_lora.yaml")
    parser.add_argument("--mode", type=str, choices=["lora", "merged"], default="merged")
    parser.add_argument("--instruction", type=str, required=True, help="Question comptable à poser au modèle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assistant = ComptaAIAssistant(config_path=args.config, mode=args.mode)
    response = assistant.generate(args.instruction)

    print("\n=== Réponse ComptaAI ===")
    print(response)


if __name__ == "__main__":
    main()
