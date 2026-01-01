# Adaptive-Intervention-for-MTL-and-other-Experiments
This repository demonstrates how a learned **Adaptive Intervention Model (AIM)** can resolve gradient interference in **Multi-Task Learning (MTL)**.  
Experiments compare **Naive MTL**, **static heuristics**, and **AIM**; results show that AIM improves robustness while providing interpretable task relationships.


---

## Repository Contents
- **`MTL-and-Experiments.ipynb`**  
  Main notebook containing all code, explanations, and visualizations.

- **`simple_mtl_model.py`**  
  Simple shared backbone with task-specific heads used in the experiments.

- **`aim_policy.py`**  
  AIM policy model implementing the adaptive intervention logic and policy network.

---

## Notebook Overview & Sections

### 1. Gradient Interference & Negative Transfer
- Visualizes opposing gradients.
- Demonstrates how conflicting gradients can harm individual tasks.
- Defines the concept of **negative transfer**.

### 2. Static Heuristics (PCGrad-style)
- Implements a projection-based heuristic to mitigate directional conflicts.
- Shows both:
  - A **success scenario**, and  
  - A **failure case** when task relationships change over time.

### 3. Adaptive Intervention Model (AIM)
- Implements the AIM training loop (Appendix A.1):
  - Compute per-task gradients
  - Feed gradients into a policy network
  - Perform differentiable gradient interventions
  - Train the policy using a held-out guidance set
- **Policy loss components**:
  - Guidance Loss  
  - Magnitude Preservation  
  - Progress Penalty  

### 4. Experiments
- **Sabotage Experiment**  
  Injects a noisy / irrelevant task and shows that AIM quarantines it, preventing corruption of the shared backbone.

- **NLP Mixed-Signals Experiment**  
  Simulates two tasks (Sentiment vs. Spam) with disjoint feature sets.  
  AIM learns to keep tasks “in their lanes” and avoids feature confusion.


> **Note:** Targets are normalized in the notebook to stabilize training and loss scales. See the notebook for implementation details.


---

## Key Findings

- AIM prevents negative transfer and can isolate noisy tasks that would otherwise harm valid tasks. 
- AIM is **interpretable**: the learned τ matrix reveals which task pairs the policy is strict about versus tolerant of.
- Static heuristics can help early training but may hinder later convergence when task relationships evolve. 

---

## Quick Comparison

| Strategy                          | Robustness | Interpretability | Notes |
|----------------------------------|------------|------------------|-------|
| Naive MTL                        | Low        | Low              | Simple sum of losses; vulnerable to noisy tasks |
| Static Heuristics (PCGrad-like)  | Medium     | Low              | Handles directional conflicts but is rigid |
| AIM (Learned Policy)             | High       | High             | Learns task-specific thresholds and balances objectives |
