# Parallelization and Scaling


---
# Parallelization Strategies

More detail at the [Ultra-Scale Playbook](huggingface.co/spaces/nanotron/ultrascale-playbook).

---
## Data Parallelism

![DP](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/dp_diagram.png)

---
## Pipeline Parallelism
![PP](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_afab.svg)
![PP](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/pp_1f1b.svg)

---
## Model Sharding
![Sharding](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/zero_memory.svg)

---
## Tensor Parallelism
![TP](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_diagram4.png)

---
## Sequence Parallelism
![SP](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/tp_sp_diagram.png)

---
## Context Parallelism
![CP](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/ring-attention.gif)

---
## Expert Parallelism
![EP](https://nanotron-ultrascale-playbook.static.hf.space/assets/images/ep_schema.png)

---
# Frameworks

---
## Native Pytorch

- Provides (D)DP and FSDP
- Wrap model
- Launch processes 

---
## Megatron-LM

- Nvidia (ROCm port exists)
- Pre-built GPT implementation
- Prototype

---
![table](https://github.com/ROCm/Megatron-LM/raw/rocm_dev/images/model_table.png)


---
- Install Megatron-LM
- Get tokenizer
- Convert and pre-process data
- Run

---
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Needed for SP

# RCCL interface
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export OMP_NUM_THREADS=1

export HSA_ENABLE_SDMA=0 # Not needed for AMD

# libfabric settings

export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

# triton
export TRITON_ALWAYS_COMPILE=1

#PARALLELISM ARGS
PP_SIZE=4
TP_SIZE=4
VPP_SIZE=2
```

---
```bash
#LLama 34B
MODEL=LLAMA-34B
NLAYERS=56
NHIDDEN=7168
NHEADS=56
FFN_HIDDEN_SIZE=20480
SEQ_LEN=4096
NUM_KV_HEADS=8
NUM_QUERY_GROUPS=8

GLOBAL_BATCH_SIZE=1024
MICRO_BATCH_SIZE=2

LR=1.5e-4
MIN_LR=1.5e-5
INIT_METHOD_STD=0.00747017

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --use-distributed-optimizer \
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --max-position-embeddings $SEQ_LEN \
    --seq-length $SEQ_LEN \
    --train-iters 10 \
    --data-path $TRAIN_DATA \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $VOCAB \
    --merge-file $MERGES \
    --bf16 \
    --disable-bias-linear \
    --init-method-std $INIT_METHOD_STD \
    --normalization RMSNorm \
    --seed 42 \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --use-rotary-position-embeddings \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    --distributed-timeout-minutes 10 \
    --no-gradient-accumulation-fusion \
    --no-bias-swiglu-fusion \
    --recompute-activations \
    $OPTIMIZER_ARGS \
    $PROFILE_ARGS \
    "
```

---
```bash
PARALLEL_ARGS="\
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel \
"

if (( VPP_SIZE > 1)); then
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --num-layers-per-virtual-pipeline-stage $VPP_SIZE"
fi
```

---
```bash
CMD=" \
    pretrain_gpt.py \
    $GPT_ARGS \
    $PARALLEL_ARGS \
    $OUTPUT_ARGS \
    --dataloader-type single \
    --num-workers 5 \
    \
    --profile-ranks 0 63 \
    --profile-step-start 3 \
    --profile-step-end 4 \
    --use-pytorch-profiler \
    --profile \
    --train-iters 5 \
    "

srun -N $Nodes \
    -n $((Nodes*8)) \
    --cpu-bind=mask_cpu:$BIND_MASK \
    --gpus=$((Nodes*8)) \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    python -u \
    $CMD |& tee train_log.log
```

---
## NeMo

- NVidia
- NeMo-run
- Requires slurm plugins
- Except for the launcher in principle compatible with LUMI

---
## HuggingFace Accelerate, PyTorch Lightning

- High-level training frameworks
- provide access to different parallelization implementations with plugins

---
## Also Existing

- TorchTitan (3D parallelism + sharding)
- Modalities (Some strategies missing, no PP)
- DeepSpeed (original model sharding from MicroSoft)
- Varuna (MicroSoft, PP + DP)
- GPTNeoX (EleutherAI, supports MI250Xs, 3D parallelism)
- OSLO (EleutherAI)
- Optimus-CC (Compressed messages)
- MosaicML/LLMFoundry


---
# Measuring Performance

---
### Metrics:
- Hardware FLOP/s
    - Depends on e.g. checkpointing or activation recomputation
    - Good for comparing to theoretical/reference performance of the hardware / finding good communication settings
- Tokens / GPU / second
    - Can not compare models of different sizes
    - Good for comparing a single model on different number of GPUs
- Model FLOP/s Utilization
    - Good for comparing frameworks and clusters
- Time to convergence
