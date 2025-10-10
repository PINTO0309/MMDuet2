# if you use multinode training, make sure these envs are set
export NNODES="${JOB_ARGS[0]}"
export NPROC_PER_NODE="${JOB_ARGS[1]}"
export GPUS_PER_NODE="${JOB_ARGS[1]}"
export MASTER_ADDR="${JOB_ARGS[2]}"
export MASTER_PORT="${JOB_ARGS[3]}"
export NODE_RANK="${JOB_ARGS[4]}"

export LLM_EVALUATOR_URL="URL"
export LLM_EVALUATOR_API_KEY="API_KEY"
export LLM_EVALUATOR_MODEL="Doubao-Seed-1.6"
export LLM_EVALUATOR_MAX_SCORES=5
export LLM_EVALUATOR_PAUC_ADD_PREV_TURNS=0
export LLM_EVALUATOR_REPETITION_IN_ALL_SPANS=1
export LLM_EVALUATOR_REPETITION_WEIGHT=2
export LLM_EVALUATOR_OUTSPAN_WEIGHT=0.5
export LLM_EVALUATOR_COMMON_PREFIX_WEIGHT=2

export VERL_AUTO_PADDING=TRUE
export RAY_IGNORE_UNHANDLED_ERRORS=1
export TP_SIZE=2

# --------------------
# 1. start multi node ray cluster
# --------------------

export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
if [ "$NODE_RANK" -eq 0 ]; then
    timestamp=$(date +"%Y%m%d-%H%M%S")
    export TENSORBOARD_DIR=tensorboard_log/${timestamp}
    mkdir $TENSORBOARD_DIR
    
    MAX_FRAMES=180
    MAX_PROMPT_LENGTH=$((MAX_FRAMES*190))
    MAX_RESP_FRAMES=90
    MAX_RESP_LENGTH=$((MAX_RESP_FRAMES*190))

    ray start --head --dashboard-host=0.0.0.0 --object-store-memory $((126*1024*1024*1024))
    sleep 30s
else
    sleep 15s
    ray start --address="$MASTER_ADDR:6379" --object-store-memory $((126*1024*1024*1024))
fi

ray status


# --------------------
# 2. start verl training
# --------------------

if [ "$NODE_RANK" -eq 0 ]; then
    export SYSTEM_PROMPT="You are a helpful assistant. Your task is to answer questions based on continuously incoming video frames. Your responses should include information from the video since your last reply (if any). If the information in this segment of the video cannot answer the question, output \"NO REPLY\"."
    mkdir ../nohup/proactive
    ckpt_path=../train/ckpt/qwen2_5/SFT_CKPT

    python -u -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=../data/live_whisperx-egoexo4d-egoexolearn0913-half_multi_half_single_question-1_sec_per_frame-20s_seg_as_example-0714-max_${MAX_FRAMES}_frames-train.parquet \
        data.val_files=../data/live_whisperx-egoexo4d-egoexolearn0913-half_multi_half_single_question-1_sec_per_frame-20s_seg_as_example-0714-max_${MAX_FRAMES}_frames-val.parquet \
        data.shuffle=False \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=./recipe/proactive/tool_config/live_whisperx-egoexo4d-egoexolearn0913-half_multi_half_single_question-20s_seg_as_example-1_sec_per_frame-max_${MAX_FRAMES}_frames.yaml \
        data.train_batch_size=$((NNODES*NPROC_PER_NODE/TP_SIZE)) \
        data.val_batch_size=$((NNODES*NPROC_PER_NODE/TP_SIZE)) \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.filter_overlong_prompts=False \
        data.return_raw_chat=True \
        data.truncation='left' \
        actor_rollout_ref.model.path=$ckpt_path \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$((NNODES*NPROC_PER_NODE/TP_SIZE)) \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH+MAX_RESP_LENGTH)) \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.005 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=$TP_SIZE \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH \
        actor_rollout_ref.rollout.response_length=$MAX_RESP_LENGTH \
        +actor_rollout_ref.rollout.max_new_tokens_per_generation=100 \
        actor_rollout_ref.rollout.temperature=1.2 \
        actor_rollout_ref.rollout.top_k=10 \
        actor_rollout_ref.rollout.do_sample=True \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_turns=$((MAX_FRAMES+10)) \
        actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
        +actor_rollout_ref.rollout.multi_turn.is_online_video_conversation=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console','tensorboard'] \
        trainer.project_name='proactive' \
        trainer.experiment_name="0814-egoexolearn0913-1_sec-outspan_penalty_0.5-common_prefix_penalty_2_rep_penalty_2-max_${MAX_FRAMES}_frames-no_freeze_vit" \
        trainer.val_before_train=False \
        trainer.validation_data_dir=../nohup/proactive-${timestamp} \
        trainer.n_gpus_per_node=$NPROC_PER_NODE \
        trainer.nnodes=$NNODES \
        trainer.save_freq=30 \
        trainer.test_freq=30 \
        trainer.total_epochs=1 \
        reward_model.launch_reward_fn_async=True \
        custom_reward_function.path=recipe/proactive/reward_function.py \
        custom_reward_function.name=pauc_reward_function \
        > ../nohup/proactive/${timestamp}-egoexolearn0913-1_sec-outspan_penalty_0.5-common_prefix_penalty_2_rep_penalty_2-max_${MAX_FRAMES}_frames-no_freeze_vit.log 2>&1

        return 0
    else
        sleep infinity
    fi
