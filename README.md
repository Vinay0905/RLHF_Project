# RLHF Continuous Agent Improvement Pipeline

A production-grade, closed-loop Reinforcement Learning from Human Feedback (RLHF) system designed to continuously train and align an autonomous LangGraph agent.

## Core Features
- **Explicit State Control**: Driven by `LangGraph`, ensuring 100% observability across all `THINK`, `ACT`, `OBSERVE`, and `RESPOND` nodes. 
- **Offline Batch Evaluation**: Safely analyzes generation logs via a `Simulated Human Evaluator` to grade behavior away from live production.
- **Dynamic Policy Gradient**: The `PolicyOptimizer` mathematically intercepts failure critiques, writes corrective AI prompts, and deploys them to `policy/vN.json`.
- **Memory Retention**: Protects against catastrophic forgetting by dynamically inheriting previous policy rules.
- **Pluggable Architecture**: Easily swap the internal `parse_mock_llm` node with live Groq, OpenAI, or Anthropic LLM endpoints natively using LangChain.

## How to Run

1. **Activate Environment**
```bash
conda activate all
```

2. **Clean State** (Optional, resets RLHF memory to baseline Gen 1)
```bash
rm -rf data policy
```

3. **Execute Pipeline**
```bash
PYTHONPATH=. python src/demo.py
```

Watch the terminal closely! You will see the agent fail a query in Generation 1, receive a bad score, be corrected by the Optimizer, and successfully improve its average reward score in Generation 2/3!