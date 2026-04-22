import os 
import json 
class Evaluator:
    """
    for simulating human evaluation and grading 
    """
    def evaluate_runs(self,generation_dir:str):
        logs_path=os.path.join(generation_dir,"traces.jsonl")
        rewards_path=os.path.join(generation_dir,"rewards.json")
        rewards=[]
        if not os.path.exists(logs_path):
            return

        with open(logs_path,'r') as f:
            for line in f:
                trace_data=json.loads(line)

                score, critique = self._score_trace(trace_data)
                rewards.append({
                    "run_id": trace_data["run_id"],
                    "score": score,
                    "critique": critique
                })

        with open(rewards_path,"w") as f:
            json.dump(rewards,f,indent=2)
        
    def _score_trace(self,trace_data):
        query = trace_data["query_text"].lower()
        trace = trace_data.get("trace", [])
        
        #netural base
        score = 3
        critique = "Neutral behavior."
        
        #langGraph state trace: Did the agent try to escalate?
        acts = [t for t in trace if t["type"] == "ACT"]
        escalations = [a for a in acts if "escalate_to_human" in a["content"]]
        
        if "order status" in query or "#12345" in query:
            # Rule 1: MUST use the check_order_status tool for order questions
            if any("check_order_status" in a["content"] for a in acts):
                score = 5
                critique = "Successfully used the correct tool for order status."
            else:
                 score = 1
                 critique = "Failed to use check_order_status tool for order query."
                 
        elif escalations:
             # Rule 2: Escalations out of laziness are punishe
             score = 2
             critique = "Escalated unnecessarily. Could have resolved via tools."
             
        elif "password" in query:
             # Rule 3: Expected KB search
             if any("search_knowledge_base" in a["content"] for a in acts):
                 score = 5
                 critique = "Correctly searched KB for standard query."
        
        return score, critique