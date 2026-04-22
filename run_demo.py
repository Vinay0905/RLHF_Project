"""
FILE: run_demo.py
DESCRIPTION: A visually polished demo script replacing basic orchestration.
Code Layout:
- run_generation(): Executes the LangGraph agent for simulated queries, runs the Evaluator to get the reward, extracts explicit Before/After critique examples, and triggers the PolicyOptimizer.
- Main Block: Iterates through 3 generations sequentially to prove RLHF improvements.
"""
import json
import os
import time

from src.settings import SETTINGS
from src.agent.graph import agent_graph
from src.agent.policy import policy_config
from src.logging.logger import TraceLogger
from src.feedback.evaluator import Evaluator
from src.training.optimizer import PolicyOptimizer

def run_generation(gen_id: int):
    print(f"\n{'='*50}\n🚀 RUNNING GENERATION {gen_id}\n{'='*50}")
    
    # Load Data
    queries_path = SETTINGS["project"]["paths"]["mock_queries"]
    with open(queries_path, "r") as f:
        queries = json.load(f)
        
    # Setup Logging
    logger = TraceLogger(gen_id)
    gen_dir = logger.generation_dir
    
    # Load Current Policy
    policy_dir = SETTINGS["project"]["paths"]["policy_dir"]
    policy_path = os.path.join(policy_dir, f"v{gen_id}.json")
    if os.path.exists(policy_path):
        policy_config.load(policy_path)
    
    print(f"[Policy Guidelines active]:")
    for g in policy_config.guidelines:
        print(f"  - {g}")
    print("-" * 50)
        
    # Production (Run Agent)
    for q in queries:
        state = {
            "run_id": f"run_{gen_id}_{int(time.time())}_{q['id']}",
            "query_id": q["id"],
            "query_text": q["text"],
            "trace": [],
            "current_thought": "",
            "tool_call": None,
            "final_response": None,
            "iteration_count": 0
        }
        final_state = agent_graph.invoke(state)
        logger.log_trace(final_state)
        
    # Feedback (Evaluate)
    print("📊 Evaluating Generation Logs...")
    evaluator = Evaluator()
    evaluator.evaluate_run(gen_dir)
    
    with open(os.path.join(gen_dir, "summary.json"), "r") as f:
        summary = json.load(f)
        avg_score = summary['average_score']
        print(f"--> Average Reward Score: {avg_score:.2f} / 5.0")
        
    # Provide Before & After Tasks Context 
    # Grab the worst query and best query from rewards to show in the demo
    rewards_path = os.path.join(gen_dir, "rewards.json")
    with open(rewards_path, 'r') as f:
        rewards = json.load(f)
        worst_run = min(rewards, key=lambda x: x["score"])
        best_run = max(rewards, key=lambda x: x["score"])
        print(f"   [Task Example - LOW SCORE ({worst_run['score']})]: {worst_run['critique']}")
        print(f"   [Task Example - HIGH SCORE ({best_run['score']})]: {best_run['critique']}")
        
    # Training / Improvement
    next_gen = gen_id + 1
    next_policy_path = os.path.join(policy_dir, f"v{next_gen}.json")
    print("🔧 Optimizing Policy for next generation based on lowest scores...")
    optimizer = PolicyOptimizer()
    optimizer.optimize(gen_dir, next_policy_path)

if __name__ == "__main__":
    policy_dir = SETTINGS["project"]["paths"]["policy_dir"]
    os.makedirs(policy_dir, exist_ok=True)
    
    # Run loop
    for i in range(1, 4):
        run_generation(i)
        time.sleep(1)
    
    print("\n✅ RLHF Pipeline Completed Successfully.")
