
import torch
import time
import numpy as np
from algoperf.workloads.wmt.wmt_pytorch.workload import WmtWorkload
from algoperf.workloads.wmt.wmt_lowrank_pytorch.workload import WmtLowRankWorkload
from algoperf import spec

def get_gpu_stats():
    if torch.cuda.is_available():
        return {
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    return {}

def run_inference_benchmark(name, model, workload, batch, rng, duration_secs=120):
    print(f"\n--- Benchmarking Inference ({name}) for {duration_secs} seconds ---")
    device = next(model.parameters()).device
    model.eval()
    
    torch.cuda.reset_peak_memory_stats()
    
    steps = 0
    latencies = []
    start_time = time.time()
    
    with torch.no_grad():
        while (time.time() - start_time) < duration_secs:
            step_start = time.time()
            
            _ = workload.model_fn(
                model, 
                batch, 
                None, 
                spec.ForwardPassMode.EVAL, 
                rng, 
                update_batch_norm=False,
                dropout_rate=0.0
            )
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            step_end = time.time()
            latencies.append(step_end - step_start)
            steps += 1

    total_time = time.time() - start_time
    avg_latency = np.mean(latencies)
    throughput = steps / total_time
    gpu_stats = get_gpu_stats()
    
    print(f"Inference Result {name}: {steps} steps in {total_time:.2f}s")
    print(f"Avg Latency: {avg_latency*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} step/s")
    
    return {
        "throughput": throughput,
        "latency_ms": avg_latency * 1000,
        "max_mem_gb": gpu_stats.get('max_allocated_gb', 0)
    }

def test_architectures():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    rng = np.array([42, 0])
    batch_size = 32
    seq_len = 256
    inputs = torch.randint(1, 32000, (batch_size, seq_len)).to(device)
    targets = torch.randint(1, 32000, (batch_size, seq_len)).to(device)
    batch = {'inputs': inputs, 'targets': targets}

    print("Initializing Workloads...")
    baseline_wl = WmtWorkload()
    baseline_model, _ = baseline_wl.init_model_fn(rng)
    baseline_model.to(device)
    
    lowrank_wl = WmtLowRankWorkload()
    lowrank_model, _ = lowrank_wl.init_model_fn(rng)
    lowrank_model.to(device)
    
    print("Compiling Models...")
    baseline_compiled = torch.compile(baseline_model)
    lowrank_compiled = torch.compile(lowrank_model)

    # Warmup
    print("Warmup (50 steps)...")
    with torch.no_grad():
        for _ in range(50):
            for model, wl in [(baseline_compiled, baseline_wl), (lowrank_compiled, lowrank_wl)]:
                _ = wl.model_fn(model, batch, None, spec.ForwardPassMode.EVAL, rng, False, dropout_rate=0.0)
    torch.cuda.synchronize()

    # Inference Benchmark (2 minutes each)
    baseline_inf = run_inference_benchmark("Baseline", baseline_compiled, baseline_wl, batch, rng, duration_secs=120)
    torch.cuda.empty_cache()
    lowrank_inf = run_inference_benchmark("Low-Rank", lowrank_compiled, lowrank_wl, batch, rng, duration_secs=120)

    # Summary
    print("\n" + "="*40)
    print("INFERENCE COMPARISON (2 min per model)")
    print("="*40)
    print(f"{'Metric':<20} | {'Baseline':<12} | {'Low-Rank':<12} | {'Diff':<10}")
    print("-" * 60)
    
    def print_row(label, b_val, l_val, format_str, is_overhead=True):
        diff = (l_val / b_val - 1) * 100 if is_overhead else (l_val - b_val)
        suffix = "%" if is_overhead else ""
        print(f"{label:<20} | {b_val:{format_str}} | {l_val:{format_str}} | {diff:+.2f}{suffix}")

    print_row("Throughput (step/s)", baseline_inf["throughput"], lowrank_inf["throughput"], ".2f", is_overhead=True)
    print_row("Avg Latency (ms)", baseline_inf["latency_ms"], lowrank_inf["latency_ms"], ".2f", is_overhead=True)
    print_row("Peak Mem (GB)", baseline_inf["max_mem_gb"], lowrank_inf["max_mem_gb"], ".2f", is_overhead=False)

if __name__ == "__main__":
    test_architectures()
