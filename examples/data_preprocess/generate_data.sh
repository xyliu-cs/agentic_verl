set -e
set -u

# Download the dataset from Hugging Face in parquet format
# Use `hf download` or `huggingface-cli download`
huggingface-cli download lindsay21/longbench_v2_transformed_rl \
  --repo-type dataset \
  --local-dir /path/to/dataset/rl_data/dataset_name