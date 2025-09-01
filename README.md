# Lyra-Mistral7B-immobilier-LoRA 🏠💶

## Description
Ce dépôt contient le travail d’adaptation **LoRA (QLoRA 4-bit)** du modèle **Mistral-7B-Instruct-v0.3**, appliqué à un cas d’usage simple et grand public :  
**l’estimation immobilière** à partir de paramètres structurés (surface, type, état, zone A/B1/B2/C).

L’objectif est de montrer la capacité à :
- créer un dataset synthétique calibré,
- entraîner et fusionner un modèle LoRA,
- publier le modèle complet sur Hugging Face,
- documenter le projet dans un format reproductible et exploitable.

---

## 🔹 Bilan technique
- **Base model** : `mistralai/Mistral-7B-Instruct-v0.3`  
- **Technique** : QLoRA (4-bit, bitsandbytes)  
- **GPU** : A100 (Colab Pro)  
- **Epochs** : 3  
- **Modules LoRA** : q_proj, k_proj, v_proj, o_proj, down_proj  
- **Fusion & export** : `merge_and_unload()` → modèle complet pushé sur Hugging Face Hub  
- **Test** : comparaison *base vs LoRA* → le LoRA produit des réponses concises et cadrées, sans hallucination de villes, respectant le format attendu.  

---

## 🔹 Valeur professionnelle
- **Vitrine MLOps légère** : pipeline complet de fine-tuning → fusion → publication → documentation.  
- **Cas d’usage accessible** : montre comment un modèle 7B peut être adapté à un secteur concret (immobilier), avec intégration possible dans un workflow (n8n / Make / API).  

---

## 🔗 Liens
- 📦 Modèle Hugging Face : [Lyra-Mistral7B-immobilier-LoRA](https://huggingface.co/jeromex1/Lyra-Mistral7B-immobilier-LoRA)  
- 📘 Autres projets Lyra (40+ repos IA & STEM) : [github.com/Jerome-openclassroom](https://github.com/Jerome-openclassroom/)  

---

## 🔹 Exemple d’utilisation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "jeromex1/Lyra-Mistral7B-immobilier-LoRA"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

system_prompt = (
    "Tu es un expert en estimation immobilière. "
    "Réponds en français, de manière concise, en 1 à 2 phrases maximum. "
    "Donne une estimation chiffrée (fourchette en euros) et un bref commentaire. "
    "Les localisations sont classées par zone : A, B1, B2 ou C. "
    "NE MENTIONNE PAS de villes ou régions, uniquement la zone donnée."
)

user_prompt = "Surface : 96 m²\nType : maison\nÉtat : bon\nZone : B1"
full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nRéponse :"

inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=80)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🔹 Exemples de résultats
| Prompt (input) | Réponse (LoRA) |
|----------------|----------------|
| Surface : 96 m²<br>Type : maison<br>État : bon<br>Zone : B1 | Valeur estimée entre 300 000 € et 400 000 €. Bien situé dans une zone B1, cette maison de 96 m² est en très bon état. |
| Surface : 55 m²<br>Type : appartement<br>État : à rénover<br>Zone : C | Estimation : 100 000 – 120 000 €<br>Commentaire : Appartement à rénover dans une zone C, l'estimation est comprise entre 100 000 et 120 000 €. |
| Surface : 120 m²<br>Type : maison<br>État : excellent<br>Zone : A | Valeur estimée entre 500 000 € et 600 000 €. Situation privilégiée, grande surface et état excellent. |

---
