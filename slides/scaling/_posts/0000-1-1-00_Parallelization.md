# Parallelization and Scaling


---
# Parallelization Strategies

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

---
## Sequence Parallelism

---
## Context Parallelism

---
## Expert Parallelism

---
# Frameworks

---
## Native Pytorch

---
## Megatron-LM

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
