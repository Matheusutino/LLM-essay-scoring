#!/bin/bash

run_and_log() {
    COMMAND=$1
    OUT_FILE=$2
    STATUS_FILE=$3

    nohup bash -c "$COMMAND; echo \$? > $STATUS_FILE" > $OUT_FILE 2>&1 &
}

mkdir -p logs
echo "Running all scripts with nohup..."

# Zero-shot
run_and_log "python -m src.scripts.evaluator_zero_shot --llm_provider 'chutes' --model_name 'Qwen/Qwen3-8B'"                            logs/zero_qwen3.out        logs/zero_qwen3.status
run_and_log "python -m src.scripts.evaluator_zero_shot --llm_provider 'chutes' --model_name 'chutesai/Mistral-Small-3.1-24B-Instruct-2503'" logs/zero_mistral.out      logs/zero_mistral.status
run_and_log "python -m src.scripts.evaluator_zero_shot --llm_provider 'chutes' --model_name 'tngtech/DeepSeek-R1T-Chimera'"                logs/zero_deepseek.out     logs/zero_deepseek.status
run_and_log "python -m src.scripts.evaluator_zero_shot --llm_provider 'chutes' --model_name 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'" logs/zero_deepseek-qwen3.out  logs/zero_deepseek-qwen3.status
run_and_log "python -m src.scripts.evaluator_zero_shot --llm_provider 'nvidia' --model_name 'nvidia/llama-3.1-nemotron-ultra-253b-v1'"      logs/zero_nemotron.out    logs/zero_nemotron.status

# Few-shot
run_and_log "python -m src.scripts.evaluator_few_shot --llm_provider 'chutes' --model_name 'Qwen/Qwen3-8B'"                                logs/few_qwen3.out         logs/few_qwen3.status
run_and_log "python -m src.scripts.evaluator_few_shot --llm_provider 'chutes' --model_name 'chutesai/Mistral-Small-3.1-24B-Instruct-2503'" logs/few_mistral.out       logs/few_mistral.status
run_and_log "python -m src.scripts.evaluator_few_shot --llm_provider 'chutes' --model_name 'tngtech/DeepSeek-R1T-Chimera'"                logs/few_deepseek.out      logs/few_deepseek.status
run_and_log "python -m src.scripts.evaluator_few_shot --llm_provider 'chutes' --model_name 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'" logs/few_deepseek-qwen3.out   logs/few_deepseek-qwen3.status
run_and_log "python -m src.scripts.evaluator_few_shot --llm_provider 'nvidia' --model_name 'nvidia/llama-3.1-nemotron-ultra-253b-v1'"      logs/few_nemotron.out     logs/few_nemotron.status

# Writer
run_and_log "python -m src.scripts.evaluator_writer --llm_provider 'chutes' --model_name 'Qwen/Qwen3-8B'"                                 logs/writer_qwen3.out      logs/writer_qwen3.status
run_and_log "python -m src.scripts.evaluator_writer --llm_provider 'chutes' --model_name 'chutesai/Mistral-Small-3.1-24B-Instruct-2503'"  logs/writer_mistral.out    logs/writer_mistral.status
run_and_log "python -m src.scripts.evaluator_writer --llm_provider 'chutes' --model_name 'tngtech/DeepSeek-R1T-Chimera'"                 logs/writer_deepseek.out   logs/writer_deepseek.status
run_and_log "python -m src.scripts.evaluator_writer --llm_provider 'chutes' --model_name 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'" logs/writer_deepseek-qwen3.out logs/writer_deepseek-qwen3.status
run_and_log "python -m src.scripts.evaluator_writer --llm_provider 'nvidia' --model_name 'nvidia/llama-3.1-nemotron-ultra-253b-v1'"       logs/writer_nemotron.out  logs/writer_nemotron.status
