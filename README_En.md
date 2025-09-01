# Lyra-Immo: A Mistral-7B-Based Real Estate Estimation Assistant

Lyra-Immo is a real estate estimation assistant project based on the fine-tuning of the Mistral-7B model. This repository contains the data, code, and instructions for training and using a model capable of estimating the value of real estate properties based on their characteristics.

## Features
- Estimation of real estate value based on area, type, condition, and location.
- Use of fine-tuning with QLoRA to adapt the Mistral-7B model for real estate estimation tasks.
- Easy integration with tools like n8n to automate estimations.

## Prerequisites
To use this project, you will need:
- Python 3.8 or higher
- Python libraries listed in `requirements.txt`
- A Hugging Face account to access the model
- Access to a GPU for training (recommended)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Lyra-Immo.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install additional libraries for fine-tuning:
   ```bash
   pip install -r requirements-finetuning.txt
   ```

## Usage
### Training the Model
To train the model, follow these steps:
1. Prepare your data in JSONL format as indicated in the `data` folder.
2. Run the training script:
   ```bash
   python train.py --model_name mistralai/Mistral-7B --dataset_path data/train.jsonl
   ```

### Using the Trained Model
To use the trained model, you can use the `inference.py` script:
   ```bash
   python inference.py --model_path path/to/model --input "Area: 96 m²
Type: house
Condition: good
Location: zone B1"
   ```

## Example Output
For an input like:
```
Area: 96 m²
Type: house
Condition: good
Location: zone B1
```
The model can return an estimation like:
```
Estimation: 264000 €
```

## Contribution
Contributions are welcome! Open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
