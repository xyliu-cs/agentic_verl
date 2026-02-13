#!/usr/bin/env bash

set -xeuo pipefail

# Increase file descriptor limit for Ray + SGLang based rollouts.
ulimit -n 65535

PROJECT_DIR=$(git rev-parse --show-toplevel)
CONFIG_ROOT=$PROJECT_DIR/examples/sglang_multiturn/config
AGENT_LOOP_CONFIG=$CONFIG_ROOT/statelm_tool_agent.yaml
TOOLS_CONFIG=$CONFIG_ROOT/tool_config/statelm_tool_optimized_config.yaml

DATA_DIR=/path/to/rl_data
TRAIN_FILE=$DATA_DIR/dataset_name/train.parquet
VAL_FILE=$DATA_DIR/dataset_name/val.parquet

WANDB_PROJECT_NAME="statelm_agentic_rl"
DATE=$(date +%Y%m%d)
WANDB_EXPERIMENT_NAME="statelm-8b-rl-lbv2-${DATE}"

MODEL_PATH=/path/to/statelm-8b
SAVE_DIR=/path/to/save_dir

MONITOR_DIR=/path/to/log_dir
LOG_FILE=$MONITOR_DIR/valid_logs/${WANDB_EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log
TRAJECTORIES_DIR=$MONITOR_DIR/trajectories/${WANDB_EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)
VAL_DATA_DIR=$MONITOR_DIR/validation_data/${WANDB_EXPERIMENT_NAME}

mkdir -p $SAVE_DIR $VAL_DATA_DIR
echo "Logging to: $LOG_FILE"
echo "Trajectories directory: $TRAJECTORIES_DIR"
echo "Validation data directory: $VAL_DATA_DIR"


# Check if the port is occupied
# sudo ss -H -ant '( sport = :59322 )'
# LISTEN_PORT=59322
# ray start --head --num-cpus=16 --dashboard-port=8265 --dashboard-host=0.0.0.0 --dashboard-agent-listen-port=$LISTEN_PORT


NNODES=2
GPUS_PER_NODE=8
TRAIN_BATCH_SIZE=32
MICRO_BATCH_SIZE=1
TOTAL_EPOCHS=2
MAX_SAMPLES_PER_TRAJECTORY=8

DUMP_TRAJECTORIES=${DUMP_TRAJECTORIES:-true}  # Set to true by default; override with 
DUMP_TRAJECTORIES_DIR=${DUMP_TRAJECTORIES_DIR:-"$TRAJECTORIES_DIR"}
DUMP_TRAJECTORIES_FREQ=${DUMP_TRAJECTORIES_FREQ:-80}  # Dump every N trajectories

# For judge model
OPENAI_API_KEY=Dummy
OPENAI_BASE_URL=http://xxx.xxx.xxx.xxx:8080/v1
OPENAI_MODEL=gpt-oss-120b

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{
  "env_vars": {
      "NCCL_IB_DISABLE": "1",
      "NCCL_P2P_DISABLE": "1",
      "NCCL_SOCKET_IFNAME": "bond1",
      "HUGGING_FACE_HUB_TOKEN": "$HUGGING_FACE_HUB_TOKEN",
      "LM_HARNESS_CACHE_PATH": "cache",
      "PYTHONUNBUFFERED": "1",
      "WANDB_API_KEY": "$WANDB_API_KEY",
      "HTTPS_PROXY": "$HTTPS_PROXY",
      "HTTP_PROXY": "$HTTP_PROXY",
      "NO_PROXY": "$NO_PROXY",
      "OPENAI_API_KEY": "Dummy",
      "OPENAI_BASE_URL": "$OPENAI_BASE_URL",
      "OPENAI_MODEL": "$OPENAI_MODEL"
  },
  "working_dir": "./",
  "pip": ["latex2sympy2", "word2number", "timeout_decorator"],
  "excludes": [".git/**"]
  }' -- PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        --config-path="$CONFIG_ROOT" \
        --config-name="statelm_tool_agent_grpo" \
        actor_rollout_ref.rollout.name=vllm \
        algorithm.adv_estimator=grpo \
        data.train_batch_size=${TRAIN_BATCH_SIZE} \
        data.max_prompt_length=1024 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.rollout.prompt_length=22528 \
        actor_rollout_ref.rollout.response_length=10240 \
        +actor_rollout_ref.rollout.multi_turn.single_turn_max_tokens=2048 \
        +actor_rollout_ref.rollout.multi_turn.max_samples_per_trajectory=${MAX_SAMPLES_PER_TRAJECTORY} \
        +actor_rollout_ref.rollout.multi_turn.downsample_mode=random \
        +actor_rollout_ref.rollout.multi_turn.model_type=qwen3 \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=80 \
        actor_rollout_ref.rollout.max_model_len=32768 \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.top_p=0.8 \
        actor_rollout_ref.rollout.top_k=20 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.default_local_dir=$SAVE_DIR/$WANDB_EXPERIMENT_NAME \
        trainer.project_name="$WANDB_PROJECT_NAME" \
        trainer.experiment_name="$WANDB_EXPERIMENT_NAME" \
        trainer.device=cuda \
        trainer.n_gpus_per_node=${GPUS_PER_NODE} \
        trainer.nnodes=${NNODES} \
        trainer.save_freq=8 \
        trainer.test_freq=8 \
        trainer.validation_data_dir="$VAL_DATA_DIR" \
        trainer.logger='["console", "wandb"]' \
        custom_reward_function.path=verl/utils/reward_score/statelm_qa.py \
        custom_reward_function.name=compute_score \
        data.train_files="$TRAIN_FILE" \
        data.val_files="$VAL_FILE" \
        trainer.total_epochs=$TOTAL_EPOCHS \
        actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 \
        actor_rollout_ref.rollout.trace.token2text=False \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOLS_CONFIG" \
        actor_rollout_ref.rollout.free_cache_engine=True \
        +actor_rollout_ref.rollout.multi_turn.exceed_length_penalty=-1.0 \
        +actor_rollout_ref.rollout.multi_turn.dump_trajectories_enabled=$DUMP_TRAJECTORIES \
        +actor_rollout_ref.rollout.multi_turn.dump_trajectories_dir="$DUMP_TRAJECTORIES_DIR" \
        +actor_rollout_ref.rollout.multi_turn.dump_trajectories_freq=$DUMP_TRAJECTORIES_FREQ \
    2>&1 | tee -a "$LOG_FILE"
