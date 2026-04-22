"""
logigng done in JSON format files .
allows appending easily

"""

import json
import os
class TraceLogger:
    def __init__(self,generation_id:int ):
        self.generation_dir=f"data/run/gen_{generation_id}"
        os.makedirs(self.generation_dir,exist_ok=True)

        self.logs_path=os.path.join(self.generation_dir,"traces.jsonl")
    def log_trace(self,state:dict):
        """
        receives the finalize agentstate
        """
        safe_state = {
            "run_id": state.get("run_id", "unknown_run"),
            "query_id": state.get("query_id", "unknown_query"),
            "query_text": state.get("query_text", ""),
            
            # The RLHF explicitly needs this to know WHAT happened
            "trace": state.get("trace", []),
            
            "final_response": state.get("final_response"),
            
            # This allows us to debug reward hacking (infinite loops)
            "iterations": state.get("iteration_count", 0)
        }
        with open(self.logs_path,"a") as f:
            f.write(json.dumps(safe_state)+'\n')