# Lyra-Mistral7B-immobilier-LoRA 🏠💶

## Description
This repository contains a **LoRA (QLoRA 4-bit)** adaptation of the **Mistral-7B-Instruct-v0.3** model, applied to a simple and accessible use case:  
**real estate estimation** based on structured parameters (surface, property type, condition, zone A/B1/B2/C).

The goal is to demonstrate the ability to:
- create a calibrated synthetic dataset,
- train and merge a LoRA model,
- publish the full model on Hugging Face,
- document the project in a reproducible and usable format.

---

## 🔹 Technical Summary
- **Base model**: `mistralai/Mistral-7B-Instruct-v0.3`  
- **Technique**: QLoRA (4-bit, bitsandbytes)  
- **GPU**: A100 (Colab Pro)  
- **Epochs**: 3  
- **LoRA modules**: q_proj, k_proj, v_proj, o_proj, down_proj  
- **Fusion & export**: `merge_and_unload()` → full model pushed to Hugging Face Hub  
- **Test**: comparison *base vs LoRA* → LoRA produces concise, structured answers without hallucinated cities, respecting the expected format.  

---

## 🔹 Professional Value
- **Light MLOps showcase**: full pipeline from fine-tuning → merge → publication → documentation.  
- **Accessible use case**: shows how a 7B model can be adapted to a concrete sector (real estate), with potential integration into workflows (n8n / Make / API).  

---

## 🔗 Links
- 📦 Hugging Face model: [Lyra-Mistral7B-immobilier-LoRA](https://huggingface.co/jeromex1/Lyra-Mistral7B-immobilier-LoRA)  
- 📘 Other Lyra projects (40+ AI & STEM repos): [github.com/Jerome-openclassroom](https://github.com/Jerome-openclassroom/)  

---

## 🔹 Usage Example

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
    "You are a real-estate estimation expert. "
    "Answer in French, concisely, in 1–2 sentences maximum. "
    "Provide a price range (in euros) and a short comment. "
    "Localisations are classified by zones: A, B1, B2, or C. "
    "NEVER mention cities or regions, only the given zone."
)

user_prompt = "Surface : 96 m²\nType : maison\nÉtat : bon\nZone : B1"
full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nRéponse :"

inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=80)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🔹 Sample Results
| Prompt (input) | LoRA Response |
|----------------|---------------|
| Surface : 96 m²<br>Type : maison<br>État : bon<br>Zone : B1 | Valeur estimée entre 300 000 € et 400 000 €. Bien situé dans une zone B1, cette maison de 96 m² est en très bon état. |
| Surface : 55 m²<br>Type : appartement<br>État : à rénover<br>Zone : C | Estimation : 100 000 – 120 000 €<br>Commentaire : Appartement à rénover dans une zone C, l'estimation est comprise entre 100 000 et 120 000 €. |
| Surface : 120 m²<br>Type : maison<br>État : excellent<br>Zone : A | Valeur estimée entre 500 000 € et 600 000 €. Situation privilégiée, grande surface et état excellent. |

---

## 💾 Repository Structure

```
lyra_transformer/
├── README.md                          # Documentation in French
├── README_en.md                       # Documentation in English
│
├── code/                              # Training and test scripts
│   └── Mistral_estate.py              # Colab script for LoRA training (real estate)
│
├── datasets/                          # Datasets used for training
│   ├── train_lyra_immo_mistral.jsonl  # Training dataset (real estate estimation)
│   └── validation_lyra_immo_mistral.jsonl # Validation dataset
│
├── curve/                             # Learning curves and metrics
│   └── learning_curve.xlsx            # Excel file (values + loss/steps graph)
```
