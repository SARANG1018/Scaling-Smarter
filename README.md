# Scaling Smarter: Scaled Dot-Product Attention and Quantized Inference for LLMs

This project explores inference-time optimization of large language models, using the open-weight **Mistral-7B** architecture. It implements and benchmarks **4-bit quantization** (via QLoRA) and **Scaled Dot Product Attention (SDPA)** to accelerate inference while maintaining model quality. Performance is measured using **CUDA kernel-level profiling**, **total inference time**, and **perplexity**.

---

## 🧠 Project Overview

Large language models (LLMs) like Mistral-7B achieve exceptional results but suffer from high latency and memory usage during inference. This project focuses on two inference-side optimizations:

- **Quantization** using 4-bit and 8-bit precision (via BitsAndBytes)
- **Attention kernel optimization** using SDPA (PyTorch fused attention backend)

Five variants were created and evaluated. The best-performing variant was fine-tuned on the *Treasure Island* dataset using parameter-efficient techniques (PEFT + LoRA) and validated on perplexity.

---

## 📊 Project Milestones

| Milestone                                       | Status     |
|------------------------------------------------|------------|
| ✅ Model Variant Construction (5 versions)      | Completed  |
| ✅ Profiling using PyTorch Profiler             | Completed  |
| ✅ Evaluation (Latency, CUDA, Perplexity)       | Completed  |
| ✅ Selection and fine-tuning of best model      | Completed  |
| ✅ Results plotted and analyzed                 | Completed  |

---

## 🧱 Repository Structure

