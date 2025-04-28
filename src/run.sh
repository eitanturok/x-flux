# set HF token

# make sure models are downloaded to workspace directory
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

python3 main.py \
    --offload \
    --model_type flux-dev-fp8 \
    --prompt "A handsome girl in a suit covered with bold tattoos and holding a pistol"
