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

- Pre-built

---
## NeMo

- NVidia supported
- Requires 

---
## HuggingFace Accelerate, PyTorch Lightning, 

---
## Also Existing

- Modalities ()
- DeepSpeed (original model sharding from MicroSoft)
- GPTNeoX ()
- OSLO ()


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
