import os 
import time 

def run_rlhf_loop(generation_id:int):
    """
    Runs one complete cycle of RLHF pipeline.
    """
    print(f"\n{'='*40}\n Starting Generation {generation_id}\n{'='*40}")
    run_dir=f"data/runs/gen_{generation_id}"
    os.makedirs(run_dir,exist_ok=True)

    print(f"Step 1: Loaded Agent Guidelines for Generation {generation_id}")
    print("Step 2: Connecting Agent to Production (Handling Queries)...")
    print("Step 3: Simulating Human Feedback...")
    print("Step 4: Training Policy Optimizer...")

if __name__=="__main__":
    for i in range(1,4):
        run_rlhf_loop(i)
        time.sleep(1)


