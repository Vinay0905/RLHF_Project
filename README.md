# RLHF-Based Continuous Agent Improvement Pipeline

This repository is my submission for the **RLHF-Based Continuous Agent Improvement Pipeline (Agentic AI System)** assignment in `Naga_Vinay_ssignment.pdf`.

The goal of the assignment was to build a production-style, agentic system that can:

- run an agent in production,
- capture reasoning and tool traces,
- generate its own feedback signals,
- improve its policy across multiple iterations,
- and redeploy the updated behavior in the next cycle.

This project implements that loop in a lightweight, fully local, mockable way using **Python + LangGraph + rule-based RLHF simulation**.

## What This Project Demonstrates

The system explicitly closes the loop:

**Production -> Trace Logging -> Feedback Evaluation -> Policy Update -> Redeployment**

In this implementation:

- the agent is modeled as an explicit LangGraph state machine with `THINK`, `ACT`, `OBSERVE`, and `RESPOND` states,
- every run is logged into a training-ready trace file,
- a simulated evaluator assigns rewards and critiques,
- an optimizer converts repeated failures into new policy rules,
- and the next generation loads the updated policy from `policy/vN.json`.

This is not “real” RL training, and that is intentional because the assignment explicitly allowed simulation as long as the control logic is correct. My focus here was to make the **agent loop explicit, observable, and easy to reason about** rather than hiding behavior inside a black-box LLM call.

## Assignment Coverage

Below is how the current project maps to the assignment requirements from the PDF.

| Assignment expectation | Status | Where it exists in this repo |
| --- | --- | --- |
| Closed-loop RLHF pipeline | Done | `src/demo.py` |
| Explicit control loops and state transitions | Done | `src/agent/graph.py`, `src/agent/state.py` |
| Agent outputs, reasoning traces, and tool logs captured | Done | `src/tracing/logger.py`, `data/run/gen_*` |
| Feedback mechanism for uncertain or critical outputs | Done | `src/feedback/evaluator.py` |
| Reward model / scoring logic | Done (simulated) | `src/feedback/evaluator.py` |
| Iterative improvement cycles | Done | `policy/v2.json`, `policy/v3.json`, `policy/v4.json` |
| 2-3 improvement cycles minimum | Done | `data/run/gen_1`, `gen_2`, `gen_3` |
| No pre-labeled dataset | Done | Feedback is generated from execution traces |
| Production-grade modular design | Done | `src/agent`, `src/feedback`, `src/training`, `src/tracing` |
| Observability and traceability | Done | `traces.jsonl` per generation |
| Architecture diagrams | Done | `Architecture_Diagrams.md`, `RLHF_Flow_Diagrams.md` |
| System design documentation | Done | `System_Design.md`, `Architecture.md` |
| Architecture decisions | Done | `Architecture_Decisions.md` |
| Trade-offs | Done | `Trade_Offs.md` |
| Failure scenarios | Done | `Failure_Scenarios.md` |
| Safety considerations | Done | `Safety_Considerations.md` |
| Evaluation methodology | Done | `Evaluation_Methodology.md` |
| No heavy external infrastructure / no external APIs | Done | Fully local mock setup |

## Current Improvement Results

The repository already includes saved run artifacts under `data/run/` and saved policy versions under `policy/`.

Using the checked-in `rewards.json` files:

| Generation | Average reward |
| --- | --- |
| Gen 1 | `3.7 / 5.0` |
| Gen 2 | `4.7 / 5.0` |
| Gen 3 | `4.7 / 5.0` |

### What improved between generations

In **Generation 1**, the agent handled order tracking and password reset questions correctly, but it made poor decisions on some policy-style queries and one package-tracking variant.

Examples of failures in Gen 1:

- it escalated refund / exchange / cancellation style questions that should have received a safe fallback,
- it missed one order-tracking phrasing (`Where is my package for #12345 right now?`) and escalated instead of using the order-status tool.

In **Generation 2**, the optimizer added stronger policy rules, including:

- always use `check_order_status` for order queries,
- do not escalate unless the user explicitly asks for a manager,
- use a safe fallback for refund / exchange / custom-order policy questions.

That change is visible both in:

- `policy/v2.json`
- and the traces in `data/run/gen_2/traces.jsonl`

By **Generation 3**, performance remained stable, which shows the updated behavior persisted after redeployment.

## How The System Works

### 1. Production agent

The agent runs as an explicit state graph:

- `THINK`: decide next step,
- `ACT`: prepare tool call,
- `OBSERVE`: capture tool output,
- `RESPOND`: finalize answer.

This logic is implemented in `src/agent/graph.py`.

### 2. Trace logging

Each run is logged with:

- query text,
- step-by-step trace,
- final response,
- iteration count.

The logger writes these records to `data/run/gen_X/traces.jsonl`.

### 3. Simulated human feedback

The evaluator reads traces offline and assigns:

- a numeric score,
- a critique explaining why the run was good or bad.

This is implemented in `src/feedback/evaluator.py`.

### 4. Policy optimization

The optimizer reads low-scoring critiques and converts them into updated behavioral rules for the next generation. Those rules are persisted as versioned policy files such as:

- `policy/v2.json`
- `policy/v3.json`
- `policy/v4.json`

This is implemented in `src/training/optimizer.py`.

### 5. Redeployment

At the start of the next generation, `src/demo.py` loads the latest policy file and runs the agent again using the updated rules.

## Why I Designed It This Way

My point of view on this assignment was:

**if the learning loop is hidden, then it is hard to trust, debug, or explain.**

Because of that, I intentionally chose:

- **explicit state transitions** over a more opaque agent wrapper,
- **offline evaluation** over live self-modifying behavior,
- **versioned JSON policies** over in-memory prompt mutation,
- **mock tools and mock feedback** so the full loop can be demonstrated locally without external dependencies,
- **modular components** so each stage of the loop can be inspected independently.

I wanted the project to feel like a small but understandable production system, not just a demo script calling an LLM repeatedly.

## Project Structure

```text
.
├── src/
│   ├── agent/        # State graph, mock LLM, policy config, tools
│   ├── feedback/     # Simulated reward model / evaluator
│   ├── training/     # Policy optimizer
│   ├── tracing/      # JSONL trace logging
│   ├── data/         # Mock input queries
│   └── demo.py       # Main closed-loop entry point
├── data/run/         # Saved traces and rewards by generation
├── policy/           # Learned policy versions
├── *.md              # Supporting architecture and evaluation docs
├── config.yaml
└── requirements.txt
```

## How To Run
### 0. Remove the data/run and the policy files for clearing the agent memory :
```bash
rm -rf data policy             
```
### 1. Install libraries :

```bash

pip install -r requirements.txt

```

### 2. Run the pipeline :

```bash

PYTHONPATH=. python3 src/demo.py

```

### 3. Inspect outputs :

After execution, inspect:

- `data/run/gen_*/traces.jsonl`
- `data/run/gen_*/rewards.json`
- `policy/v*.json`

## Key Files To Review

If you want the quickest technical walkthrough, I recommend reviewing these in order:

1. `src/demo.py`
2. `src/agent/graph.py`
3. `src/feedback/evaluator.py`
4. `src/training/optimizer.py`
5. `data/run/gen_1` -> `gen_3`
6. `policy/v2.json` -> `v4.json`

## Supporting Documentation

- [Architecture](Architecture.md)
- [Architecture Decisions](Architecture_Decisions.md)
- [Architecture Diagrams](Architecture_Diagrams.md)
- [System Design](System_Design.md)
- [RLHF Flow Diagrams](RLHF_Flow_Diagrams.md)
- [Evaluation Methodology](Evaluation_Methodology.md)
- [Trade-Offs](Trade_Offs.md)
- [Failure Scenarios](Failure_Scenarios.md)
- [Safety Considerations](Safety_Considerations.md)

## Honest Limitations

This project is intentionally a **simulated RLHF system**, so a few things are simplified:

- the evaluator is heuristic and rule-based rather than a learned reward model,
- the “policy update” is symbolic rule injection rather than parameter optimization,
- the tools are mocked,
- the agent reasoning is deterministic rather than driven by a live model,
- improvement plateaus after the main failure modes are corrected.

I consider those acceptable trade-offs for this assignment because the main objective was to demonstrate **control-loop design, state management, observability, feedback integration, and iterative improvement logic**.

## Final Note

<!-- If I were extending this beyond the assignment, my next steps would be:

- add broader query diversity and harder edge cases,
- introduce confidence-aware routing,
- separate evaluator features from decision rules more cleanly,
- add automated tests for evaluator and optimizer behavior,
- and replace the deterministic mock policy with a more realistic model adapter while preserving the same explicit control loop. -->

This repository is therefore both:

- a working assignment submission,
- and a foundation for a more realistic RLHF-style agent improvement system.
