# Range Pair Model

## Setup
- Install dependencies: `pip install -r requirements.txt`

## Run
- Start the script: `python range_pair_model.py`
- When prompted, enter queries like `give between 1 to 1000`
- Type `exit` to quit

## What It Does
- Loads a small set of sample prompts with their numeric ranges
- Trains a TF-IDF based matcher on those prompts
- Matches your query to the trained examples and returns the stored ranges
- Uses Ollama locally (if available) for better intent matching; falls back to ML model otherwise

