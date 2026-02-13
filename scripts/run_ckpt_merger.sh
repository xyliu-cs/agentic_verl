set -e

model_path=/path/to/model/global_step_xxx/actor

target_path=${model_path}/huggingface

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${model_path} \
    --target_dir ${target_path}