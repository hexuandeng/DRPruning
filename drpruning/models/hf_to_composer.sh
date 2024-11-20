# Define the Hugging Face model name and the output path
HF_MODEL_NAME=../LLMs/${1}
OUTPUT_PATH=drpruning/models/${1}-composer.pt

# Create the necessary directory if it doesn't exist
mkdir -p $(dirname $OUTPUT_PATH)

# Convert the Hugging Face model to Composer key format
python3 -m drpruning.utils.hf_to_composer $HF_MODEL_NAME $OUTPUT_PATH
