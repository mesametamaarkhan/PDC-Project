# 🔄 Parallel Framework for Dynamic SSSP Updates

A high-performance, architecture-independent framework for maintaining **Single Source Shortest Path (SSSP)** trees in **dynamic graphs**. This project addresses the inefficiencies of traditional static SSSP algorithms when applied to real-world, constantly changing networks.

---

## 📌 Key Contributions

- **Problem Addressed:** Traditional SSSP algorithms are inefficient for large dynamic networks due to full recomputation on each change.
- **Proposed Solution:** A parallel update framework that incrementally adjusts only the affected regions of the graph.
- **Architecture Independence:** Designed to run on both shared-memory CPUs and GPUs.
- **Performance-Oriented:** Minimizes synchronization overhead and uses hardware-specific optimizations.

---

## 🧠 Algorithm Overview

### Step 1: Identify Affected Subgraph
- For **insertions**, check if the new edge offers a shorter path.
- For **deletions**, mark child vertices of removed edges if part of the current SSSP.

### Step 2: Incrementally Update SSSP
- Iterative distance updates without global synchronization locks.
- Prevents cycles and ensures convergence to the correct shortest paths.

---

## 🏗️ Implementation Design

### CPU (Shared-Memory)
- Built using **OpenMP**.
- Supports **asynchronous updates**, **batch processing**, and adjustable update granularity.

### GPU (CUDA)
- Uses **CSR format** and **Vertex Marking Functional Blocks (VMFB)**.
- Reduces atomic operations and handles updates using smart flag-based logic.

---

## 📊 Experimental Evaluation

- **Datasets:** Real-world (e.g., Orkut, LiveJournal) and synthetic (e.g., RMAT24_G) graphs.
- **Scenarios:** Benchmarked with up to 100M edge updates.
- **Performance:**
  - Up to **8.5× speedup** over Gunrock (GPU baseline).
  - Up to **5× speedup** over Galois (CPU baseline).
- **Scalability:** Performs well across increasing core counts and large update batches.
- **Limitations:** Degrades in highly deletion-heavy scenarios (e.g., >85% affected vertices).

---

## 📐 Theoretical Analysis

- **Correctness:** Proven to maintain valid shortest-path trees and prevent cycles.
- **Complexity:**
  - **Step 1:** `O(m/p)` — with `m` = edge updates, `p` = processors.
  - **Step 2:** `O(Dxd/p)` — with `x` = affected vertices, `d` = average degree, `D` = diameter.

---

## 🧩 Roadmap & Future Work

- Explore a **hybrid strategy**: dynamically choose between update and full recomputation.
- **Predictive planning**: Optimize updates based on anticipated changes.
- Study **non-random or structured change patterns** for smarter optimization.

---

## 📁 Repository Structure (Coming Soon)

- `docs/` – Project documentation and theoretical details
- `presentation/` – Project presentation slides and visuals
- `src/` – Source code for CPU and GPU implementations *(planned)*
- `datasets/` – Benchmark datasets *(planned)*

---

## 📄 Current Status

✅ Documentation  
✅ Presentation  
🕓 Source Code – Coming Soon  
🕓 Datasets – Coming Soon  

---

## 📬 Contact

For questions, discussions, or collaborations, feel free to open an issue or reach out via the repository discussions tab.

---

