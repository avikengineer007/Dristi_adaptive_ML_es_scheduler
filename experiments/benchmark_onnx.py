import json
import os
import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None


class ActorPolicyModule(nn.Module):
    """Clean PyTorch forward module extracting the actor head from SB3 PPO."""

    def __init__(self, policy):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.mlp_extractor(obs)
        logits = self.action_net(features)
        return logits


def export_ppo_models(
    model_path: str = "models/ppo_pure.zip",
    onnx_path: str = "models/ppo_policy.onnx",
    pt_path: str = "models/ppo_policy.pt",
) -> Dict[str, str]:
    """
    Exports SB3 PPO actor policy network to TorchScript and ONNX formats
    for high-speed real-time deployment.
    """
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    model = PPO.load(model_path)
    policy = model.policy.to("cpu")
    actor = ActorPolicyModule(policy)
    actor.eval()

    obs_dim = model.observation_space.shape[0]
    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)

    exported = {}

    # 1. Export TorchScript (Zero-dependency embedded C++ deployable)
    traced = torch.jit.trace(actor, dummy_input)
    traced.save(pt_path)
    print(f"Exported PPO policy network to TorchScript: {pt_path}")
    exported["torchscript"] = pt_path

    # 2. Export ONNX (if onnx package is available)
    if onnx is not None:
        try:
            torch.onnx.export(
                actor,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["observation"],
                output_names=["action_logits"],
                dynamic_axes={"observation": {0: "batch_size"}, "action_logits": {0: "batch_size"}},
                dynamo=False,
            )
            print(f"Exported PPO policy network to ONNX: {onnx_path}")
            exported["onnx"] = onnx_path
        except Exception as e:
            print(f"ONNX export deferred ({e}). Using TorchScript deployment artifact.")

    return exported


def benchmark_inference_latency(
    model_path: str = "models/ppo_policy.pt",
    obs_dim: int = 65,
    num_iterations: int = 1000,
    output_json: str = "results/onnx_latency_benchmark.json",
) -> Dict[str, Any]:
    """
    Benchmarks CPU inference latency per scheduling decision,
    verifying compliance with the 1.0 ms real-time EW slot deadline.
    """
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    latencies_us = []

    if model_path.endswith(".onnx") and ort is not None:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        for _ in range(50):
            dummy = np.random.uniform(0.0, 1.0, size=(1, obs_dim)).astype(np.float32)
            _ = session.run(None, {input_name: dummy})

        for _ in range(num_iterations):
            inp = np.random.uniform(0.0, 1.0, size=(1, obs_dim)).astype(np.float32)
            t0 = time.perf_counter()
            _ = session.run(None, {input_name: inp})
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1e6)
        backend = "ONNX Runtime (CPU)"
    else:
        # High-performance TorchScript JIT model
        pt_model = torch.jit.load(model_path)
        pt_model.eval()

        with torch.no_grad():
            for _ in range(50):
                dummy = torch.randn(1, obs_dim)
                _ = pt_model(dummy)

            for _ in range(num_iterations):
                inp = torch.randn(1, obs_dim)
                t0 = time.perf_counter()
                _ = pt_model(inp)
                t1 = time.perf_counter()
                latencies_us.append((t1 - t0) * 1e6)
        backend = "TorchScript JIT (CPU)"

    mean_us = float(np.mean(latencies_us))
    p50_us = float(np.percentile(latencies_us, 50))
    p95_us = float(np.percentile(latencies_us, 95))
    p99_us = float(np.percentile(latencies_us, 99))
    max_us = float(np.max(latencies_us))

    results = {
        "backend": backend,
        "deployed_model": model_path,
        "iterations": num_iterations,
        "mean_latency_microseconds": round(mean_us, 2),
        "p50_latency_microseconds": round(p50_us, 2),
        "p95_latency_microseconds": round(p95_us, 2),
        "p99_latency_microseconds": round(p99_us, 2),
        "max_latency_microseconds": round(max_us, 2),
        "realtime_slot_deadline_ms": 1.0,
        "is_realtime_compliant": bool(max_us < 1000.0),
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=======================================================")
    print(f" CPU REAL-TIME INFERENCE LATENCY BENCHMARK")
    print(f" Backend: {backend} | Iterations: {num_iterations}")
    print("=======================================================")
    print(f" Mean Latency:  {mean_us:.2f} µs ({mean_us/1000.0:.4f} ms)")
    print(f" p50 Latency:   {p50_us:.2f} µs ({p50_us/1000.0:.4f} ms)")
    print(f" p95 Latency:   {p95_us:.2f} µs ({p95_us/1000.0:.4f} ms)")
    print(f" p99 Latency:   {p99_us:.2f} µs ({p99_us/1000.0:.4f} ms)")
    print(f" Real-time Slot Deadline: 1.0 ms (1000 µs)")
    print(f" Real-time Compliant: {'PASS' if results['is_realtime_compliant'] else 'FAIL'}")
    print(f" Results exported to: {output_json}\n")

    return results


if __name__ == "__main__":
    if os.path.exists("models/ppo_pure.zip"):
        exported = export_ppo_models("models/ppo_pure.zip")
        model_to_bench = exported.get("onnx", exported.get("torchscript", "models/ppo_policy.pt"))
        benchmark_inference_latency(model_to_bench)
    else:
        print("models/ppo_pure.zip not found. Run experiments/train_ppo.py first.")
