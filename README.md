# Scaling Smarter: Scaled Dot-Product Attention and Quantized Inference for LLMs

This project explores inference-time optimization of large language models, using the open-weight **Mistral-7B** architecture. It implements and benchmarks **4-bit quantization** (via QLoRA) and **Scaled Dot Product Attention (SDPA)** to accelerate inference while maintaining model quality. Performance is measured using **CUDA kernel-level profiling**, **total inference time**, and **perplexity**.

---

## Project Overview

Large language models (LLMs) like Mistral-7B achieve exceptional results but suffer from high latency and memory usage during inference. This project focuses on two inference-side optimizations:

- **Quantization** using 4-bit and 8-bit precision (via BitsAndBytes)
- **Attention kernel optimization** using SDPA (PyTorch fused attention backend)

Five variants were created and evaluated. The best-performing variant was fine-tuned on the *Treasure Island* dataset using parameter-efficient techniques (PEFT + LoRA) and validated using perplexity.

---

## Project Milestones

| Milestone                                  | Status     |
|-------------------------------------------|------------|
| Model Variant Construction (5 versions) | Completed  |
| Profiling using PyTorch Profiler        | Completed  |
| Evaluation (Latency, CUDA, Perplexity)  | Completed  |
| Selection and fine-tuning of best model | Completed  |
| Results plotted and analyzed            | Completed  |

---

## Repository Structure

```
├── scalingsmarter_hpml_inference_final.ipynb     # Complete workflow with profiling + plots
├── sdpa_4b_quantized_model/                      # Stores final fine-tuned model checkpoints (saved locally)
├── images/                                       # Includes all plots used in report
├── cuda_op_times_inference.json                  # Raw CUDA profiler dump
├── perplexities_all.json                         # Precomputed perplexity scores
├── README.md                                     # current file
├── treasure_island.txt                           # test file
```

---

## How to Run

Ensure the following packages are installed:

```bash
pip install transformers accelerate bitsandbytes torch
```

Then launch the notebook:

```bash
jupyter notebook scalingsmarter_hpml_inference_final.ipynb
```

To test the final fine-tuned model with a prompt:

```python
prompt = "<s>[INST] Tell me a pirate story with a treasure map and an island. [/INST]"
# Run model.generate() and decode response
```

---

## Results and Observations

### 🔹 Stage 1: Variant Benchmarking

Before benchmarking, each variant was run through a standardized pipeline that generated the model response, computed perplexity, profiled CUDA kernels, and stored the metrics to a `.json` file.

**Example Output Snapshot (Quantized 4-bit model with FlashAttention):**

- **Response Perplexity**: 195.97  
- **Self CUDA Time**: 6.648s  
- **Top Kernels**: `kgemm_4bit_inference`, `MatMul4Bit`, `aten::mm`

![Profiler Output for 4-bit Quantized Inference](images/profiler_snapshot_4bit.png)  
*Profiler output for 4-bit quantized inference*

---

- **Lowest Perplexity**: 4-Bit Quantized Mistral Model (196.42)
- **Fastest CUDA Inference Time**: 4-Bit Quantized Mistral Model (6.6 ms)
- **Visualizations**:

![Perplexity Comparison](images/perplexity_comparision_stage1.png)  
`perplexity_comparision_stage1.png`

![CUDA vs CPU Inference Time](images/cuda_vrs_cpu_model_stage1.png)  
`cuda_vrs_cpu_model_stage1.png`

### 🔹 Stage 2: Kernel Profiling
- **Best CUDA Kernel Efficiency**: 4-Bit Quantized Mistral Model (0.0049 ms/op)
- **Visualizations**:

![Total Inference Time](images/model_inference_stage2.png)

  - `model_inference_stage2.png`

![CUDA Time per Operation](images/cudatime_per_op_stage2.png)

  - `cudatime_per_op_stage2.png`

### 🔹 Stage 3: Final Fine-Tuning
- **Training Setup**: LoRA + QLoRA (4-bit) with SDPA
- **Dataset**: 10% subset of *Treasure Island*
- **Final Perplexity (Evaluated)**: 10.65

---

## Key Takeaways

- Combining **4-bit quantization** and **scaled dot-product attention (SDPA)** significantly reduces inference time without sacrificing generation quality.
- **Kernel-level profiling** enabled precise selection of the most execution-efficient model.
- The final fine-tuned model remains **lightweight** and suitable for **deployment on consumer-grade GPUs**.

---

## Credits

**Base Model**: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

**Frameworks Used**:
- PyTorch
- Hugging Face Transformers
- BitsAndBytes (4-bit quantization)
- PEFT (LoRA)

**Profiling Tools**:
- PyTorch Profiler
- JSON Trace Exports

---

## License

MIT License. Open for academic and non-commercial research use.
