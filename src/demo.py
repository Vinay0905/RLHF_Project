import json 
import os 
import time 

from src.settings import SETTINGS
from src.agent.graph import agent_graph
from src.agent.policy import policy_config
from src.tracing.logger import TraceLogger
from src.feedback.evaluator import Evaluator
from src.training.optimizer import PolicyOptimizer

def run_generation(gen_id:int):
    print(f"\n{'='*50}\n running generation {gen_id}\n{'='*50}")
    #deployement, loading policy dynamicakly to check if its updated or not 
    policy_dir=SETTINGS["project"]["paths"]["policy_dir"]
    policy_path = os.path.join(policy_dir, f"v{gen_id}.json")
    if os.path.exists(policy_path):
        with open(policy_path, "r") as f:
            data = json.load(f)
            policy_config.guidelines = data["guidelines"]
            policy_config.escalation_threshold = data["escalation_threshold"]

    print("[Active Agent Guidelines]:    ")
    for rule in policy_config.guidelines:
        print(f" --> {rule}")
    print("-"*50)

    #production
    print("Processing queries")
    queries_path = SETTINGS["project"]["paths"]["mock_queries"]
    with open(queries_path, "r") as f:
        queries = json.load(f)
        
    logger = TraceLogger(gen_id)
    
    for q in queries:
        state = {
            "run_id": f"run_{gen_id}_{q['id']}",
            "query_id": q["id"],
            "query_text": q["text"],
            "trace": [],
            "current_thought": "",
            "tool_call": None,
            "final_response": None,
            "iteration_count": 0
        }
        # Run our Explicit State Machine
        final_state = agent_graph.invoke(state)
        logger.log_trace(final_state)

    #feedback
    print("\n Human Feedback Supervisor evaluating logs...")
    evaluator = Evaluator()
    evaluator.evaluate_runs(logger.generation_dir)
    
    # Wait for file to write
    time.sleep(0.5) 
    
    # Print the requested metrics 
    rewards_path = os.path.join(logger.generation_dir, "rewards.json")
    if os.path.exists(rewards_path):
        with open(rewards_path, 'r') as f:
            rewards = json.load(f)
            avg_score = sum(r["score"] for r in rewards) / len(rewards)
            print(f"★ Average Reward Score: {avg_score:.2f} / 5.0")
            
            # Print a "Before vs After" critique example for the interviewer
            worst_run = min(rewards, key=lambda x: x["score"])
            best_run = max(rewards, key=lambda x: x["score"])
            print(f"   [Task Example - WORST CRITIQUE]: {worst_run['critique']}")
            print(f"   [Task Example - BEST CRITIQUE]: {best_run['critique']}")
        
    # --- TRAINING/OPTIMIZER ---
    print("\n Optimizing Policy to fix the worst failures...")
    next_gen = gen_id + 1
    next_policy_path = os.path.join(policy_dir, f"v{next_gen}.json")
    
    optimizer = PolicyOptimizer()
    optimizer.optimize(logger.generation_dir, next_policy_path,policy_path)


if __name__ == "__main__":
    # Create the policy folder if it doesn't exist

    os.makedirs(SETTINGS["project"]["paths"]["policy_dir"], exist_ok=True)
    
    # Run the closed loop for exactly 3 iterations as requested

    for i in range(1, 4):
        run_generation(i)
        time.sleep(2)
        
    print("\n RLHF Pipeline Completed Successfully.")

    
    