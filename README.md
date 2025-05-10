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

├── scalingsmarter_hpml_inference_final.ipynb # Complete workflow with profiling + plots
├── sdpa_4b_quantized_model/ # Stores final fine-tuned model checkpoints (saved locally)
├── images/ # Includes all plots used in report
├── cuda_op_times_inference.json # Raw CUDA profiler dump
├── perplexities_all.json # Precomputed perplexity scores
├── README.md # This file

---

## ⚙️ How to Run

Make sure the following packages are installed:

```bash
pip install transformers accelerate bitsandbytes torch
Then run the notebook:

bash
Copy
Edit
jupyter notebook scalingsmarter_hpml_inference_final.ipynb
To test the final fine-tuned model with a prompt:

python
Copy
Edit
prompt = "<s>[INST] Tell me a pirate story with a treasure map and an island. [/INST]"
# Run model.generate() and decode
📈 Results and Observations
Stage 1: Variant Benchmarking
Lowest Perplexity: 4-Bit Quantized Model (196.42)

Fastest CUDA Inference: 4-Bit Quantized Model (6.6 ms)

Graph: perplexity_comparision_stage1.png, cuda_vrs_cpu_model_stage1.png

Stage 2: Kernel Profiling
Best CUDA Efficiency: 4-Bit Quantized Model (0.0049 ms/op)

Graph: model_inference_stage2.png, cudatime_per_op_stage2.png

Stage 3: Final Fine-Tuning
Training Approach: LoRA + QLoRA (4-bit) + SDPA

Dataset: 10% of Treasure Island

Final Perplexity: 10.65

📌 Key Takeaways
Combining SDPA and 4-bit QLoRA compression yields faster inference without sacrificing generation quality.

CUDA profiling at kernel-level granularity helped select the most hardware-efficient model.

The final fine-tuned model is lightweight and deployable on consumer GPUs.

🔗 Credits
Model: Mistral-7B-Instruct-v0.2

Libraries: PyTorch, HuggingFace Transformers, BitsAndBytes, LoRA (PEFT)

Profiler: PyTorch Profiler + JSON Export

🏁 License
MIT License. Academic use encouraged.
