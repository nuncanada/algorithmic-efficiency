
import torch
import time
import pandas as pd
import math
import random
from algoperf.workloads.finewebedu_lm.finewebedu_lm_pytorch.experimental_models import ExperimentalTransformer, ExperimentalConfig

def run_experiment(name, cfg_params):
    print(f"Running: {name}...", end=" ", flush=True)
    config = ExperimentalConfig(
        vocab_size=50257,
        seq_len=64, 
        model_dim=128,
        expanded_model_dim=512,
        num_layers=4,
        num_heads=4,
        **cfg_params
    )
    
    model = ExperimentalTransformer(config)
    if torch.cuda.is_available():
        model = model.cuda()
    
    try:
        compiled_model = torch.compile(model)
        
        bsz = 2
        x = torch.randint(0, config.vocab_size, (bsz, config.seq_len))
        targets = torch.randint(0, config.vocab_size, (bsz, config.seq_len))
        if torch.cuda.is_available():
            x, targets = x.cuda(), targets.cuda()
        
        # Warmup
        for _ in range(2):
            compiled_model(x, targets)
        
        start_time = time.time()
        total_loss = 0
        num_steps = 5
        for _ in range(num_steps):
            _, loss = compiled_model(x, targets)
            total_loss += loss.item()
        
        avg_time = (time.time() - start_time) / num_steps
        avg_loss = total_loss / num_steps
        ppl = math.exp(avg_loss) if avg_loss < 20 else 1e9
        
        num_params = sum(p.numel() for p in model.parameters())
        print("Done.")
        
        return {
            "Experiment": name,
            "Loss": round(avg_loss, 4),
            "PPL": round(ppl, 2),
            "Time/Step": round(avg_time, 4),
            "Params": num_params,
            "IntAttnDim": config.internal_attn_dim if config.internal_attn_dim else config.model_dim,
            "Rec": config.recursion_steps,
            "Ouro": config.use_ouroboros,
            "DepTensor": config.use_dependency_tensor,
            "LatPart": config.use_latent_partitioning
        }
    except Exception as e:
        print(f"Failed: {e}")
        return None

results = []

# 1. Independent Ideas
independent_tests = [
    ("Baseline", {}),
    ("Ouroboros", {"use_ouroboros": True, "recursion_steps": 2}),
    ("DepTensor", {"use_dependency_tensor": True}),
    ("LatentPart", {"use_latent_partitioning": True}),
    ("InterpAttn (Small)", {"internal_attn_dim": 64}),
]

for name, params in independent_tests:
    res = run_experiment(name, params)
    if res: results.append(res)

# 2. Random Combinations (Mix)
all_flags = {
    "use_ouroboros": [True, False],
    "use_dependency_tensor": [True, False],
    "use_latent_partitioning": [True, False],
    "recursion_steps": [1, 2, 3],
    "internal_attn_dim": [None, 64, 192]
}

print("\nStarting randomized combinations...")
for i in range(5):
    combo = {k: random.choice(v) for k, v in all_flags.items()}
    res = run_experiment(f"Combo_{i}", combo)
    if res: results.append(res)

df = pd.DataFrame(results)
print("\n--- FINAL RESEARCH TABLE ---")
print(df.to_markdown())
df.to_csv("research_results.csv", index=False)
