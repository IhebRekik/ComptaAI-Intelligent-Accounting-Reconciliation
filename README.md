# ComptaAI

**ComptaAI** est un assistant IA spécialisé en comptabilité, obtenu par fine-tuning LoRA du modèle **Qwen3-8B** à l'aide de [Unsloth](https://github.com/unslothai/unsloth) et de [TRL](https://github.com/huggingface/trl).

Le modèle est entraîné pour :
- Répondre à des questions comptables
- Expliquer des écritures comptables
- Assister au lettrage comptable
- Analyser des opérations financières

## 📁 Structure du projet

```
ComptaAI/
├── configs/
│   └── qwen_lora.yaml       # Configuration unique du projet
├── dataset/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── src/
│   ├── train.py              # Script principal d'entraînement (LoRA + SFTTrainer)
│   ├── dataset.py            # Chargement, validation et préparation des données
│   ├── prompt.py             # Construction des prompts (build_training_prompt)
│   ├── utils.py               # Fonctions utilitaires génériques
│   ├── evaluation.py         # Évaluation post-entraînement
│   └── inference.py          # Utilisation du modèle en production
├── checkpoints/              # Checkpoints d'entraînement (généré)
├── logs/                     # Logs TensorBoard (généré)
├── lora/                     # Adaptateurs LoRA entraînés (généré)
├── merged_model/             # Modèle fusionné autonome (généré)
├── outputs/                  # Résultats d'évaluation, exports (généré)
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Prérequis

- Python 3.11+
- GPU CUDA (testé sur Google Colab, GPU Tesla T4 gratuit)
- Voir `requirements.txt` pour les dépendances complètes

## 🚀 Installation

```bash
git clone https://github.com/<votre-compte>/ComptaAI.git
cd ComptaAI
pip install -r requirements.txt
```

## 📝 Préparer les données

Les jeux de données sont au format **JSONL**, un objet JSON par ligne avec deux champs obligatoires :

```json
{"instruction": "Qu'est-ce que le lettrage comptable ?", "response": "Le lettrage comptable consiste à ..."}
```

Placez vos fichiers dans `dataset/train.jsonl`, `dataset/validation.jsonl` et `dataset/test.jsonl` (les chemins sont configurables dans `configs/qwen_lora.yaml`).

## 🔧 Configuration

Tous les paramètres du projet (modèle, données, LoRA, hyperparamètres d'entraînement, répertoires, génération) sont centralisés dans `configs/qwen_lora.yaml`. Modifiez ce fichier pour adapter le projet à vos besoins sans toucher au code.

## 🏋️ Entraînement

```bash
cd src
python train.py --config ../configs/qwen_lora.yaml
```

Pour reprendre un entraînement interrompu à partir du dernier checkpoint :

```bash
python train.py --config ../configs/qwen_lora.yaml --resume_from_checkpoint ../checkpoints/checkpoint-100
```

Le suivi de l'entraînement (perte, taux d'apprentissage, utilisation GPU) est disponible via TensorBoard :

```bash
tensorboard --logdir ../logs
```

## 📊 Évaluation

```bash
python evaluation.py --config ../configs/qwen_lora.yaml --mode lora
# ou
python evaluation.py --config ../configs/qwen_lora.yaml --mode merged
```

Les résultats (perte, perplexité) sont sauvegardés dans `outputs/` au format JSON et CSV.

## 💬 Inférence

En ligne de commande :

```bash
python inference.py --config ../configs/qwen_lora.yaml --mode merged \
  --instruction "Comment lettrer une facture partiellement payée ?"
```

En Python :

```python
from inference import ComptaAIAssistant

assistant = ComptaAIAssistant(config_path="../configs/qwen_lora.yaml", mode="merged")
reponse = assistant.generate("Comment comptabiliser une note de frais ?")
print(reponse)
```

## 🧩 Architecture technique

- **Modèle de base** : Qwen3-8B, chargé en quantification 4 bits via `FastLanguageModel.from_pretrained()` (Unsloth)
- **Fine-tuning** : LoRA (PEFT), appliqué sur les couches `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` — seuls ces adaptateurs sont entraînés, le modèle de base reste figé
- **Boucle d'entraînement** : `SFTTrainer` (TRL), avec accumulation de gradients, scheduler cosine et précision mixte
- **Suivi** : TensorBoard, checkpoints réguliers avec reprise automatique

## 📄 Licence

Projet à but pédagogique / professionnel. Adaptez cette section selon la licence choisie.
"# ComptaAI-Intelligent-Accounting-Reconciliation" 
