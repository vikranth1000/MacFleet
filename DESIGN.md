# MacFleet: Distributed Training Across Apple Silicon Macs

> **Design Document v1.0** — This document is the single source of truth for the project. AI coding assistants (Cursor, Claude Code, etc.) should read this before making any implementation decisions.

---

## 1. Project Overview

### 1.1 What Is This?

MacFleet is a Python framework that turns multiple Apple Silicon MacBooks into a distributed training cluster connected via Thunderbolt 4. It enables data-parallel and pipeline-parallel training of PyTorch models across heterogeneous Mac hardware, with automatic workload balancing based on each node's compute and memory capacity.

### 1.2 Why Does This Matter?

- **No existing solution** does distributed training natively across Macs using Thunderbolt. PyTorch's distributed package targets NCCL (NVIDIA) or Gloo (CPU-only, no MPS awareness).
- **Apple Silicon** has unified memory (CPU+GPU share RAM) and fast MPS backend — but these advantages are locked to single machines today.
- **Thunderbolt 4** provides ~40 Gbps bandwidth between Macs — 40x faster than WiFi and competitive with InfiniBand in small clusters.
- The framework is designed to be **open-source, pip-installable**, and usable by anyone with 2+ Macs.

### 1.3 Target Outcomes

| Goal | Metric |
|------|--------|
| Functional distributed training | Train ResNet-50 on CIFAR-10 across 2 Macs, converge to >93% accuracy |
| Measurable speedup | ≥1.5x wall-clock speedup vs single Mac for compute-bound models |
| Gradient compression | Demonstrate <50% communication overhead with Top-K + FP16 compression |
| Fault tolerance | Training resumes within 10s if one node disconnects and reconnects |
| Scalability design | Architecture supports N nodes (validated with 2, designed for more) |
| Open-source ready | pip-installable, documented, with examples and benchmarks |

---

## 2. Hardware Constraints (CRITICAL — Read This First)

### 2.1 Node Specifications

| Spec | Node 0 (Master) — MacBook Pro | Node 1 (Worker) — MacBook Air |
|------|-------------------------------|-------------------------------|
| Chip | Apple M4 Pro | Apple M4 |
| CPU Cores | 12 (10P + 2E) | 10 (4P + 6E) |
| GPU Cores | 16 | 10 |
| Neural Engine | 16-core | 16-core |
| Unified RAM | 24 GB | 16 GB |
| Memory Bandwidth | ~273 GB/s | ~120 GB/s |
| Available Storage | ~300 GB | ~60 GB |
| Thunderbolt 4 | Yes (USB-C) | Yes (USB-C) |
| MPS Support | Yes (PyTorch ≥2.0) | Yes (PyTorch ≥2.0) |

### 2.2 Key Asymmetries That Drive Design Decisions

1. **Memory asymmetry (24 GB vs 16 GB)**: The MacBook Air is the bottleneck. Any model that must fit entirely on one node cannot exceed ~10 GB of GPU memory (leaving headroom for OS, PyTorch overhead, activations, gradients). This rules out any model >~3B parameters at FP16 on the Air, and >~2B at FP32.

2. **Compute asymmetry (16 GPU cores vs 10 GPU cores)**: The Pro has ~1.6x the GPU throughput. Naive data parallelism (50/50 split) would waste Pro's capacity waiting for Air. **We must implement weighted data splitting** — Pro gets ~62% of each batch, Air gets ~38%.

3. **Storage asymmetry (300 GB vs 60 GB)**: Datasets must live on the Pro. The Air should stream data from Pro or use a small local cache. Checkpoints should be saved to Pro.

4. **Thermal asymmetry**: The Air is passively cooled (no fan). Under sustained GPU load, it will thermal throttle. The framework must detect throughput drops and dynamically rebalance.

### 2.3 Memory Budget Per Node

```
MacBook Pro (24 GB total):
├── macOS + background     ~4 GB
├── PyTorch + Python        ~2 GB
├── Model weights (FP16)    variable
├── Optimizer states         2x model size (Adam)
├── Activations/gradients   variable
└── Available for ML        ~18 GB usable

MacBook Air (16 GB total):
├── macOS + background      ~4 GB
├── PyTorch + Python         ~2 GB
├── Model weights (FP16)     variable
├── Optimizer states          2x model size (Adam)
├── Activations/gradients    variable
└── Available for ML         ~10 GB usable
```

### 2.4 Interconnect: Thunderbolt 4 Bridge

- **Bandwidth**: 40 Gbps (theoretical), ~32 Gbps practical = ~4 GB/s
- **Latency**: <1ms round-trip over TB bridge
- **Setup**: macOS System Settings → Network → Thunderbolt Bridge. Both Macs get link-local IPs (169.254.x.x) or static IPs (e.g., 10.0.0.1 / 10.0.0.2).
- **Comparison**: A ResNet-50 has ~25M params = 50 MB at FP16. At 4 GB/s, that's ~12.5ms to transmit all gradients. This is fast enough for real speedup.

---

## 3. Architecture

### 3.1 High-Level System Diagram

```
┌──────────────────────────────────┐     Thunderbolt 4      ┌──────────────────────────────────┐
│       NODE 0 (MacBook Pro)       │    (~4 GB/s, <1ms)     │       NODE 1 (MacBook Air)       │
│            MASTER                │◄──────────────────────►│            WORKER                │
│                                  │                         │                                  │
│  ┌───────────────────────────┐   │                         │  ┌───────────────────────────┐   │
│  │     Training Engine       │   │                         │  │     Training Engine       │   │
│  │  ┌─────────┐ ┌─────────┐ │   │                         │  │  ┌─────────┐ ┌─────────┐ │   │
│  │  │ Forward │→│Backward │ │   │                         │  │  │ Forward │→│Backward │ │   │
│  │  │  Pass   │ │  Pass   │ │   │                         │  │  │  Pass   │ │  Pass   │ │   │
│  │  └─────────┘ └────┬────┘ │   │                         │  │  └─────────┘ └────┬────┘ │   │
│  │              Gradients    │   │                         │  │              Gradients    │   │
│  │                   │       │   │                         │  │                   │       │   │
│  │            ┌──────▼─────┐ │   │                         │  │            ┌──────▼─────┐ │   │
│  │            │  Gradient  │ │   │                         │  │            │  Gradient  │ │   │
│  │            │ Compressor │ │   │                         │  │            │ Compressor │ │   │
│  │            │ (TopK+FP16)│ │   │                         │  │            │ (TopK+FP16)│ │   │
│  │            └──────┬─────┘ │   │                         │  │            └──────┬─────┘ │   │
│  └───────────────────┼───────┘   │                         │  └───────────────────┼───────┘   │
│                      │           │                         │                      │           │
│  ┌───────────────────▼───────┐   │                         │  ┌───────────────────▼───────┐   │
│  │   Communication Layer     │   │                         │  │   Communication Layer     │   │
│  │   (gRPC over TB4)        │◄──┼────────────────────────►│  │   (gRPC over TB4)        │   │
│  │   - AllReduce             │   │                         │  │   - AllReduce             │   │
│  │   - Broadcast             │   │                         │  │   - Broadcast             │   │
│  │   - Scatter/Gather        │   │                         │  │   - Scatter/Gather        │   │
│  └───────────────────────────┘   │                         │  └───────────────────────────┘   │
│                                  │                         │                                  │
│  ┌───────────────────────────┐   │                         │  ┌───────────────────────────┐   │
│  │   Node Services           │   │                         │  │   Node Services           │   │
│  │   - Health Monitor        │   │                         │  │   - Health Monitor        │   │
│  │   - Data Server (master)  │   │                         │  │   - Thermal Monitor       │   │
│  │   - Checkpoint Manager    │   │                         │  │   - Data Client           │   │
│  │   - Cluster Coordinator   │   │                         │  │   - Workload Reporter     │   │
│  └───────────────────────────┘   │                         │  └───────────────────────────┘   │
└──────────────────────────────────┘                         └──────────────────────────────────┘
```

### 3.2 Component Breakdown

#### A. Cluster Coordinator (Master Only)
- Runs on Node 0 (MacBook Pro)
- Handles node registration, heartbeats, and cluster state
- Assigns ranks and workload weights to each node
- Initiates training runs and manages distributed checkpointing
- Serves as the data source (dataset lives on Pro's SSD)

#### B. Communication Layer (Both Nodes)
- **Transport**: gRPC over Thunderbolt 4 bridge IP
- **Serialization**: Raw tensor bytes (not protobuf for tensors — too slow). Use gRPC for control messages, raw TCP sockets for tensor data.
- **Collective Operations**:
  - `allreduce(tensor)` — average gradients across nodes (primary operation)
  - `broadcast(tensor, src)` — send model weights from master to workers
  - `scatter(tensor, src)` — distribute data batches
- **Why gRPC + raw TCP hybrid**: gRPC is great for structured control messages (register, heartbeat, sync barriers). But protobuf serialization of large float tensors is 5-10x slower than raw bytes. So tensor data goes over raw TCP sockets on a separate port.

#### C. Gradient Compressor (Both Nodes)
- **Top-K Sparsification**: Only transmit the top 10% of gradient values (by magnitude). Accumulate residuals locally (error feedback) to prevent information loss.
- **FP16 Quantization**: Cast FP32 gradients to FP16 before transmission, cast back after AllReduce.
- **Combined**: Top-10% + FP16 reduces communication volume by ~20x.
- These are configurable — user can disable for debugging.

#### D. Training Engine (Both Nodes)
- Wraps any `torch.nn.Module`
- Handles forward/backward pass on MPS device
- Calls gradient compressor after backward pass
- Calls communication layer for AllReduce
- Applies optimizer step after receiving aggregated gradients
- **Weighted data parallelism**: Master assigns batch proportions based on each node's throughput (calibrated at startup)

#### E. Health & Thermal Monitor (Both Nodes)
- Polls `powermetrics` or `IOKit` for thermal state
- Reports throughput (samples/sec) to coordinator every N steps
- Coordinator rebalances workload if throughput ratio shifts >15%

#### F. Data Distribution
- Dataset lives on Node 0 (MacBook Pro) — it has the storage
- Node 1 (MacBook Air) receives mini-batches over TB4 at training time
- Alternative: Pre-shard dataset and copy a shard to Air's SSD for faster I/O (preferred for image datasets)
- For small datasets (<5 GB): copy to both nodes at startup
- For large datasets (>5 GB): stream from Pro to Air

#### G. Checkpoint Manager (Master)
- Saves model weights + optimizer state to Pro's SSD
- Saves cluster state (which node had which data indices) for resumability
- Checkpoints every N epochs or on-demand
- On failure recovery: broadcasts latest checkpoint to reconnected node

---

## 4. Training Modes

### 4.1 Data Parallel (Primary — Implement First)

Each node holds a **full copy** of the model. Each node processes a different subset of the training batch. After backward pass, gradients are averaged via AllReduce.

```
Step-by-step:
1. Master broadcasts model weights to all nodes (only at init or after recovery)
2. Master splits batch: Pro gets 62% of samples, Air gets 38% (weighted by calibrated throughput)
3. Each node: forward pass → loss → backward pass (all on MPS)
4. Each node: compress gradients (Top-K + FP16)
5. AllReduce: exchange compressed gradients, decompress, average
6. Each node: optimizer.step() with averaged gradients
7. Sync barrier → next step
```

**Model size limit**: Must fit on the Air (~10 GB usable) = ~3B params at FP16, ~1.5B at FP32. Practically, stick to models ≤1B params for data parallel.

### 4.2 Pipeline Parallel (Stretch Goal — Design Now, Implement If Time Allows)

Model is split by layers across nodes. Each node holds a **shard** of the model. Micro-batches flow through nodes in a pipeline.

```
Pro (layers 0-N/2) → sends activations → Air (layers N/2-N) → sends gradients back
```

**Advantage**: Can train models larger than single-node memory (up to ~28 GB combined, ~20 GB usable).
**Challenge**: Pipeline bubbles reduce efficiency. Implement GPipe-style micro-batching to mitigate.

**Use pipeline parallel when**: Model doesn't fit on the Air (>10 GB), or when model is so large it benefits from splitting.

---

## 5. Technology Stack

### 5.1 Core Dependencies

| Component | Technology | Reason |
|-----------|-----------|--------|
| ML Framework | PyTorch ≥2.1 | MPS backend support, autograd, model zoo |
| GPU Backend | MPS (Metal Performance Shaders) | Only GPU option on Apple Silicon |
| Tensor Comm | Raw TCP sockets (asyncio) | Fastest for large tensor transfers — no serialization overhead |
| Control Comm | gRPC + protobuf | Structured messages (heartbeat, register, sync) with built-in streaming |
| Discovery | Bonjour/zeroconf | Auto-discover Macs on same network; also works over TB bridge |
| Compression | NumPy / custom | Top-K sparsification, FP16 casting |
| CLI | Click or Typer | User-facing launch commands |
| Monitoring | Rich (terminal UI) | Real-time training dashboard |
| Testing | pytest | Unit and integration tests |
| Packaging | pip (pyproject.toml) | pip-installable distribution |

### 5.2 Python Version

- Python 3.11+ (required for best MPS + PyTorch compatibility on macOS)

### 5.3 System Requirements

- macOS 14.0+ (Sonoma) or 15.0+ (Sequoia) — needed for MPS stability
- Xcode Command Line Tools (for Metal compilation)
- Thunderbolt 4 cable (for inter-node communication)

---

## 6. Project Structure

```
macfleet/
├── pyproject.toml               # Package config, dependencies, entry points
├── README.md                    # User-facing docs with quickstart
├── DESIGN.md                    # This file
├── LICENSE                      # MIT
│
├── macfleet/                    # Main package
│   ├── __init__.py              # Version, public API exports
│   │
│   ├── core/                    # Core distributed primitives
│   │   ├── __init__.py
│   │   ├── coordinator.py       # Cluster coordinator (master node)
│   │   ├── worker.py            # Worker node logic
│   │   ├── node.py              # Base node class (shared by coordinator/worker)
│   │   ├── cluster_state.py     # Cluster topology, node registry, health state
│   │   └── config.py            # Configuration dataclasses (ClusterConfig, NodeConfig, TrainingConfig)
│   │
│   ├── comm/                    # Communication layer
│   │   ├── __init__.py
│   │   ├── transport.py         # Raw TCP async tensor transport (send_tensor, recv_tensor)
│   │   ├── grpc_service.py      # gRPC server/client for control messages
│   │   ├── collectives.py       # AllReduce, Broadcast, Scatter, Gather implementations
│   │   ├── proto/               # Protobuf definitions
│   │   │   ├── control.proto    # RegisterNode, Heartbeat, SyncBarrier, StartTraining
│   │   │   └── control_pb2.py   # Generated
│   │   └── discovery.py         # Bonjour/zeroconf node discovery
│   │
│   ├── compression/             # Gradient compression
│   │   ├── __init__.py
│   │   ├── topk.py              # Top-K sparsification with error feedback
│   │   ├── quantize.py          # FP32→FP16 quantization for transmission
│   │   └── pipeline.py          # Composable compression pipeline (TopK → FP16 → send → FP16→FP32 → decompress)
│   │
│   ├── training/                # Training engines
│   │   ├── __init__.py
│   │   ├── data_parallel.py     # DistributedDataParallel wrapper
│   │   ├── pipeline_parallel.py # PipelineParallel wrapper (stretch goal)
│   │   ├── distributed_sampler.py # Weighted distributed data sampler
│   │   ├── trainer.py           # High-level Trainer class (user-facing API)
│   │   └── checkpoint.py        # Distributed checkpoint save/load
│   │
│   ├── monitoring/              # Health and performance monitoring
│   │   ├── __init__.py
│   │   ├── health.py            # Heartbeat, node liveness detection
│   │   ├── thermal.py           # macOS thermal state monitoring
│   │   ├── throughput.py        # Samples/sec tracking, dynamic rebalancing trigger
│   │   └── dashboard.py         # Rich terminal UI for training progress
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── tensor_utils.py      # Tensor serialization (to/from bytes), device transfer helpers
│       ├── network.py           # IP detection, TB bridge detection, port management
│       └── logging.py           # Structured logging setup
│
├── cli/                         # Command-line interface
│   ├── __init__.py
│   └── main.py                  # `macfleet launch`, `macfleet status`, `macfleet benchmark`
│
├── examples/                    # Example training scripts
│   ├── train_resnet.py          # ResNet-50 on CIFAR-10 (primary demo)
│   ├── train_gpt2.py           # GPT-2 (124M) fine-tuning on text
│   └── train_vit.py            # Vision Transformer on ImageNet subset
│
├── benchmarks/                  # Performance measurement
│   ├── bandwidth_test.py        # TB4 throughput measurement
│   ├── allreduce_bench.py       # AllReduce latency vs tensor size
│   ├── compression_bench.py     # Compression ratio vs accuracy impact
│   └── scaling_bench.py         # 1-node vs 2-node speedup curves
│
└── tests/                       # Test suite
    ├── test_transport.py        # TCP tensor send/recv
    ├── test_collectives.py      # AllReduce correctness
    ├── test_compression.py      # Compression fidelity
    ├── test_sampler.py          # Weighted sampler distribution
    └── test_integration.py      # End-to-end 2-node training (needs both Macs)
```

---

## 7. Implementation Plan (2-3 Week MVP)

### Phase 1: Foundation (Days 1-4)

**Goal**: Two Macs can discover each other, send tensors back and forth over TB4.

| Task | File(s) | Details |
|------|---------|---------|
| TB4 bridge setup | README.md | Document manual setup: System Settings → Network → Thunderbolt Bridge. Assign static IPs: 10.0.0.1 (Pro), 10.0.0.2 (Air) |
| Configuration | `core/config.py` | `ClusterConfig`, `NodeConfig`, `TrainingConfig` dataclasses. All params in one place. |
| Tensor transport | `comm/transport.py` | Async TCP server/client. `send_tensor(tensor, addr)` and `recv_tensor()`. Protocol: 4-byte header (tensor size) + raw bytes. Use `asyncio` for non-blocking I/O. |
| Bandwidth benchmark | `benchmarks/bandwidth_test.py` | Send tensors of varying sizes (1MB to 500MB), measure throughput. Target: >2 GB/s sustained. |
| Node discovery | `comm/discovery.py` | Bonjour/zeroconf registration: each node publishes `_macfleet._tcp.local.` service. Fallback: manual IP in config. |
| gRPC control plane | `comm/grpc_service.py`, `proto/control.proto` | Messages: `RegisterNode(hostname, ip, rank, gpu_cores, ram_gb)`, `Heartbeat(rank, timestamp, throughput)`, `SyncBarrier(step)`, `StartTraining(config)` |
| Coordinator & worker | `core/coordinator.py`, `core/worker.py` | Coordinator accepts registrations, assigns ranks. Worker connects, registers, waits for commands. |

**Phase 1 Deliverable**: Run `macfleet launch --role master` on Pro and `macfleet launch --role worker --master 10.0.0.1` on Air. They connect, exchange a test tensor, print confirmation.

### Phase 2: Data Parallel Training (Days 5-10)

**Goal**: Train ResNet-50 on CIFAR-10 across both Macs with gradient synchronization.

| Task | File(s) | Details |
|------|---------|---------|
| AllReduce | `comm/collectives.py` | For 2 nodes: direct exchange (each sends its gradients to the other, both average). Design interface for Ring-AllReduce (N>2) but implement direct first. |
| Gradient compression | `compression/topk.py`, `compression/quantize.py` | Top-K: keep top 10% by magnitude, accumulate residuals. FP16: cast before send, cast back after receive. |
| Compression pipeline | `compression/pipeline.py` | `CompressPipeline([TopKCompressor(ratio=0.1), FP16Quantizer()])`. Composable, configurable. |
| Weighted sampler | `training/distributed_sampler.py` | Extends PyTorch's `DistributedSampler`. Splits indices by weight ratio (default: calibrated from throughput). Pro gets more samples. |
| DDP wrapper | `training/data_parallel.py` | `MacFleetDDP(model)`. Registers backward hooks on each parameter to trigger AllReduce after gradient computation. |
| Trainer | `training/trainer.py` | High-level API: `trainer = Trainer(model, optimizer, train_loader, config)`, `trainer.fit(epochs=10)`. Handles the full loop. |
| Throughput calibration | `monitoring/throughput.py` | At startup, each node runs 10 forward+backward passes on a dummy batch. Reports samples/sec to coordinator. Coordinator computes weight ratio. |
| ResNet example | `examples/train_resnet.py` | Complete training script. <30 lines of user code. |

**Phase 2 Deliverable**: `python examples/train_resnet.py --distributed` trains on both Macs. Loss converges. Print per-step timing breakdown (compute vs communication).

### Phase 3: Robustness & Polish (Days 11-16)

**Goal**: Fault tolerance, monitoring, benchmarks, and documentation.

| Task | File(s) | Details |
|------|---------|---------|
| Health monitoring | `monitoring/health.py` | Heartbeat every 2s. If no heartbeat for 6s, mark node as dead. Master continues training alone. When node comes back: broadcast latest weights, resume distributed. |
| Thermal monitoring | `monitoring/thermal.py` | Poll `sudo powermetrics --samplers smc -i 1000 -n 1` for thermal pressure. If Air reports "heavy" thermal pressure, reduce its batch proportion by 20%. |
| Checkpointing | `training/checkpoint.py` | Save every N epochs. Save: model state_dict, optimizer state_dict, epoch, step, sampler state, compression residuals. Load and resume seamlessly. |
| Dashboard | `monitoring/dashboard.py` | Rich terminal table showing: loss, throughput, communication time, compression ratio, node health, thermal state. Updates every step. |
| Communication overlap | `training/data_parallel.py` | Overlap gradient AllReduce with backward pass of earlier layers. Use PyTorch's `register_comm_hook`. This is an optimization — implement if time allows, otherwise document as future work. |
| GPT-2 example | `examples/train_gpt2.py` | Fine-tune GPT-2 (124M) on a text dataset. Shows the framework works for NLP too. |
| Benchmarks | `benchmarks/` | Generate plots: (1) throughput vs tensor size, (2) 1-node vs 2-node training time, (3) compression ratio vs accuracy, (4) scaling efficiency. |
| README & docs | `README.md` | Quickstart, architecture diagram, benchmark results, API reference. |
| Package | `pyproject.toml` | Make pip-installable: `pip install macfleet`. Entry point: `macfleet` CLI. |

**Phase 3 Deliverable**: Polished, documented, benchmarked, pip-installable package. Ready for GitHub release and portfolio.

---

## 8. User-Facing API Design

### 8.1 CLI

```bash
# On MacBook Pro (master):
macfleet launch --role master --port 50051

# On MacBook Air (worker):
macfleet launch --role worker --master 10.0.0.1:50051

# Check cluster status:
macfleet status

# Run bandwidth benchmark:
macfleet benchmark --type bandwidth

# Run training benchmark:
macfleet benchmark --type training --model resnet50
```

### 8.2 Python API (what users write in training scripts)

```python
import torch
import torchvision
from macfleet import Trainer, ClusterConfig, TrainingConfig

# 1. Define model (standard PyTorch)
model = torchvision.models.resnet50()

# 2. Define training config
training_config = TrainingConfig(
    epochs=10,
    batch_size=128,              # Total batch size (split across nodes by weight)
    learning_rate=0.1,
    compression="topk_fp16",     # Options: "none", "topk", "fp16", "topk_fp16"
    topk_ratio=0.1,              # Keep top 10% of gradients
    checkpoint_every=2,          # Checkpoint every 2 epochs
)

# 3. Define cluster config
cluster_config = ClusterConfig(
    role="master",               # or "worker"
    master_addr="10.0.0.1",
    master_port=50051,
    tensor_port=50052,           # Separate port for raw tensor transfers
)

# 4. Standard PyTorch dataset/loader
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
    transform=torchvision.transforms.ToTensor())

# 5. Create trainer and run
trainer = Trainer(
    model=model,
    dataset=dataset,
    training_config=training_config,
    cluster_config=cluster_config,
    optimizer_cls=torch.optim.SGD,
    optimizer_kwargs={"momentum": 0.9, "weight_decay": 1e-4},
)

trainer.fit()  # Handles everything: distribution, compression, sync, checkpointing
```

### 8.3 API Design Principles

1. **Zero changes to model code** — any `torch.nn.Module` works as-is.
2. **Minimal user code** — the training script should be <30 lines. Framework handles all distributed complexity.
3. **Graceful degradation** — if only one node is available, training proceeds on that node alone (single-node mode). No code changes needed.
4. **Configurable everything** — compression, batch weighting, checkpoint frequency, etc. Sensible defaults for all.

---

## 9. Communication Protocol Specification

### 9.1 Control Channel (gRPC)

```protobuf
service ClusterControl {
    rpc Register(RegisterRequest) returns (RegisterResponse);
    rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
    rpc SyncBarrier(BarrierRequest) returns (BarrierResponse);
    rpc StartTraining(TrainingConfigProto) returns (Ack);
    rpc StopTraining(StopRequest) returns (Ack);
    rpc GetClusterState(Empty) returns (ClusterStateProto);
}

message RegisterRequest {
    string hostname = 1;
    string ip_address = 2;
    int32 gpu_cores = 3;
    int32 ram_gb = 4;
    float memory_bandwidth_gbps = 5;
    int32 tensor_port = 6;        // Port for raw tensor transfers
}

message RegisterResponse {
    int32 assigned_rank = 1;
    float workload_weight = 2;    // 0.0-1.0, proportion of batch
    int32 world_size = 3;
}

message HeartbeatRequest {
    int32 rank = 1;
    float throughput_samples_per_sec = 2;
    string thermal_state = 3;     // "nominal", "fair", "serious", "critical"
    int64 timestamp_ms = 4;
}
```

### 9.2 Tensor Channel (Raw TCP)

```
┌─────────────────────────────────────────────────────┐
│ Header (16 bytes)                                    │
│ ┌──────────┬──────────┬──────────┬────────────────┐ │
│ │ msg_type │ dtype    │ n_dims   │ payload_size   │ │
│ │ (4B)     │ (4B)     │ (4B)     │ (4B)           │ │
│ └──────────┴──────────┴──────────┴────────────────┘ │
├─────────────────────────────────────────────────────┤
│ Shape (n_dims × 4 bytes)                             │
│ e.g., [64, 3, 224, 224] → 16 bytes                  │
├─────────────────────────────────────────────────────┤
│ Payload (raw tensor bytes)                           │
│ Size = payload_size bytes                            │
└─────────────────────────────────────────────────────┘

msg_type values:
  0x01 = TENSOR_GRADIENT     (AllReduce)
  0x02 = TENSOR_WEIGHTS      (Broadcast)
  0x03 = TENSOR_ACTIVATIONS  (Pipeline parallel)
  0x04 = COMPRESSED_GRADIENT (Sparse + quantized)

For COMPRESSED_GRADIENT (0x04):
┌──────────────────────────────────────────────────────┐
│ Header (16 bytes) — same as above                     │
├──────────────────────────────────────────────────────┤
│ Compression metadata (12 bytes)                       │
│ ┌────────────┬──────────────┬──────────────────────┐ │
│ │ orig_numel │ topk_count   │ orig_dtype           │ │
│ │ (4B)       │ (4B)         │ (4B)                 │ │
│ └────────────┴──────────────┴──────────────────────┘ │
├──────────────────────────────────────────────────────┤
│ Indices (topk_count × 4 bytes) — int32               │
├──────────────────────────────────────────────────────┤
│ Values (topk_count × 2 bytes) — float16              │
└──────────────────────────────────────────────────────┘
```

### 9.3 AllReduce Protocol (2-node direct exchange)

```
Node 0                                  Node 1
  │                                        │
  │──── compressed_gradients_0 ──────────►│
  │◄──── compressed_gradients_1 ──────────│
  │                                        │
  │ decompress(grad_1)                     │ decompress(grad_0)
  │ local_grad = (grad_0 + grad_1) / 2    │ local_grad = (grad_0 + grad_1) / 2
  │                                        │
  │ optimizer.step(local_grad)             │ optimizer.step(local_grad)
```

For N>2 nodes (future): implement Ring-AllReduce where each node sends 1/N of its gradients to the next node in a ring, completing in N-1 steps.

---

## 10. Gradient Compression Details

### 10.1 Top-K Sparsification with Error Feedback

```
Algorithm:
  residual = zeros_like(gradient)  # Persistent per-parameter

  Each step:
    1. accumulated = gradient + residual
    2. topk_indices = top_k_by_magnitude(accumulated, k=numel*ratio)
    3. topk_values = accumulated[topk_indices]
    4. residual = accumulated
    5. residual[topk_indices] = 0    # Remove what we're sending
    6. Send (topk_indices, topk_values)
```

**Why error feedback matters**: Without it, the 90% of gradients we drop each step are lost forever. With error feedback, they accumulate and eventually get sent when they become large enough. Mathematically, this preserves convergence guarantees.

**Ratio**: Start with 0.1 (keep top 10%). This gives ~10x reduction in communication. Can tune from 0.01 to 1.0.

### 10.2 FP16 Quantization

```
Send side: values_fp16 = topk_values.half()    # FP32 → FP16 (2x reduction)
Recv side: values_fp32 = values_fp16.float()    # FP16 → FP32 (restore)
```

**Combined**: Top-10% keeps 10% of values, FP16 halves each value's size = **~20x total compression**.

For a ResNet-50 (25M params, 100MB gradients at FP32):
- After Top-10%: 2.5M values = 10MB
- After FP16: 2.5M × 2 bytes = 5MB values + 2.5M × 4 bytes indices = 15MB total
- Transfer time at 4 GB/s: ~3.75ms
- Without compression: 100MB / 4 GB/s = 25ms

---

## 11. Benchmark Plan

### 11.1 Metrics to Measure

| Metric | How | Target |
|--------|-----|--------|
| TB4 bandwidth | Send random tensors of varying sizes | >2 GB/s sustained |
| AllReduce latency | Time a full allreduce for ResNet-50 gradients | <15ms (compressed), <30ms (uncompressed) |
| Training throughput (single) | Train ResNet-50 on Pro alone, measure samples/sec | Baseline |
| Training throughput (distributed) | Train ResNet-50 on Pro+Air, measure samples/sec | ≥1.5x baseline |
| Communication overhead | (allreduce_time / total_step_time) × 100 | <30% with compression |
| Scaling efficiency | distributed_throughput / (N × single_throughput) | >75% for 2 nodes |
| Convergence | Final accuracy: single vs distributed | Within 0.5% of each other |
| Compression impact | Accuracy with vs without compression | Within 1% of uncompressed |

### 11.2 Benchmark Models

| Model | Params | Size (FP16) | Purpose |
|-------|--------|-------------|---------|
| ResNet-50 | 25M | 50 MB | Primary benchmark — compute-heavy CNN |
| GPT-2 (124M) | 124M | 248 MB | NLP benchmark — transformer architecture |
| ViT-Base | 86M | 172 MB | Vision transformer — modern architecture |

### 11.3 Output

Generate charts (matplotlib) for:
1. **Throughput vs Tensor Size**: Shows TB4 bandwidth saturation point
2. **Training Speedup**: Bar chart — 1 node vs 2 nodes, with/without compression
3. **Communication Breakdown**: Stacked bar — compute time vs compress time vs transfer time vs decompress time
4. **Convergence Curves**: Loss over time — single node vs distributed, overlaid
5. **Scaling Efficiency**: Line chart — ideal linear scaling vs actual (designed for N=1,2, extrapolated for 3,4)

---

## 12. Error Handling & Fault Tolerance

### 12.1 Failure Modes & Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Worker disconnects (cable unplugged) | No heartbeat for 6s | Master continues single-node training. Logs warning. |
| Worker reconnects | Worker sends Register again | Master broadcasts latest model weights. Worker rejoins training at current step. |
| Worker thermal throttle | Heartbeat reports "serious" thermal state | Master reduces worker's batch proportion by 30%. Rebalances after 60s of "nominal". |
| Master crashes | Worker detects no heartbeat | Worker saves local checkpoint, exits gracefully. Training resumes on restart. |
| Network congestion | AllReduce takes >5x baseline | Log warning. If persistent, reduce compression ratio (send less data). |
| OOM on Air | Caught in try/except | Reduce batch size for Air by 50%, log warning, continue. |

### 12.2 Graceful Degradation Ladder

```
Normal:  Pro (62%) + Air (38%) = 100% distributed throughput
    ↓ Air thermal throttle
Degraded: Pro (75%) + Air (25%) = ~85% distributed throughput
    ↓ Air disconnects
Single:  Pro (100%) = ~65% of distributed throughput (still faster than no framework)
    ↓ Air reconnects
Recovery: Broadcast weights → Resume distributed training
```

---

## 13. Important Implementation Notes

### 13.1 MPS Device Quirks

- **MPS ↔ CPU transfers are needed for communication**: Gradients live on MPS. Before sending over network, copy to CPU (`tensor.cpu()`). After receiving, move back to MPS (`tensor.to("mps")`). This adds overhead but is unavoidable.
- **MPS doesn't support all ops**: Some PyTorch operations fall back to CPU on MPS. If a model has unsupported ops, the framework should catch and warn, not crash.
- **Use `torch.mps.synchronize()`**: MPS operations are async. Always synchronize before timing or sending tensors to ensure computation is complete.
- **Memory management**: Call `torch.mps.empty_cache()` periodically to prevent MPS memory fragmentation, especially on the Air.

### 13.2 Thunderbolt Bridge Setup (Must Document for Users)

```bash
# On BOTH Macs:
# 1. Connect Thunderbolt 4 cable
# 2. System Settings → Network → Thunderbolt Bridge
# 3. Configure IPv4: Manually
#    - Mac Pro: IP 10.0.0.1, Subnet 255.255.255.0
#    - Mac Air: IP 10.0.0.2, Subnet 255.255.255.0
# 4. Verify: ping 10.0.0.2 from Pro, ping 10.0.0.1 from Air
```

### 13.3 Performance-Critical Code Paths

These are the hot paths that must be optimized:

1. **Tensor serialization** (`tensor_utils.py`): Use `tensor.numpy().tobytes()` for CPU tensors. Avoid pickle, avoid protobuf for tensor data.
2. **Top-K selection** (`topk.py`): Use `torch.topk()` on GPU (MPS) — do NOT copy to CPU first.
3. **TCP send/recv** (`transport.py`): Use `asyncio` with large buffer sizes (at least 1MB). Use `sendall()` to avoid partial sends.
4. **AllReduce** (`collectives.py`): Overlap send and receive using async — send your gradients while receiving the other node's.

### 13.4 What NOT to Build

- **No custom CUDA/Metal kernels** — PyTorch's MPS backend and torch ops are sufficient
- **No Kubernetes/Docker** — this runs bare-metal on macOS
- **No web UI** — terminal dashboard (Rich) is sufficient for MVP
- **No multi-tenancy** — one training job at a time
- **No automatic hyperparameter tuning** — out of scope
- **No data preprocessing pipeline** — use standard PyTorch DataLoader

---

## 14. Testing Strategy

### 14.1 Unit Tests (Run on Single Mac)

| Test | What It Validates |
|------|-------------------|
| `test_transport.py` | Send/recv tensors via loopback (127.0.0.1). Verify exact byte equality. |
| `test_compression.py` | Top-K keeps correct number of values. FP16 round-trip within tolerance. Error feedback accumulates correctly. |
| `test_sampler.py` | Weighted sampler distributes indices correctly. No overlap between nodes. All indices covered. |
| `test_collectives.py` | AllReduce on loopback. Two async tasks simulate two nodes. Result = average of inputs. |

### 14.2 Integration Tests (Require Both Macs)

| Test | What It Validates |
|------|-------------------|
| `test_integration.py::test_two_node_allreduce` | Real AllReduce over TB4. Verify both nodes get same result. |
| `test_integration.py::test_training_convergence` | Train 5 epochs of ResNet on CIFAR-10. Loss decreases. Accuracy improves. |
| `test_integration.py::test_node_failure_recovery` | Kill worker mid-training. Master continues. Restart worker. Worker resyncs. |

---

## 15. Repository & Release Checklist

### 15.1 GitHub Repository Setup

- [ ] MIT License
- [ ] `.gitignore` (Python, macOS, PyTorch checkpoints, datasets)
- [ ] `pyproject.toml` with all dependencies
- [ ] GitHub Actions CI (unit tests on macOS runner)
- [ ] Releases with changelog

### 15.2 README Structure

```
# MacFleet 🖥️⚡🖥️
> Distributed ML training across Apple Silicon Macs over Thunderbolt

## Why?
[1 paragraph + benchmark chart]

## Quickstart
[5-step setup: install, connect cable, configure IPs, launch, train]

## Architecture
[System diagram from this doc]

## Benchmarks
[Charts from Section 11]

## API
[Code example from Section 8.2]

## Supported Models
[Table of tested models + sizes]

## Roadmap
[What's next: N-node, pipeline parallel, Metal kernels]
```

### 15.3 What "Done" Looks Like

1. ✅ `pip install macfleet` works
2. ✅ Two Macs train ResNet-50 together, converge to >93% CIFAR-10 accuracy
3. ✅ Measurable speedup (≥1.5x) with benchmark charts to prove it
4. ✅ Gradient compression demonstrably reduces communication overhead
5. ✅ Training survives a node disconnect and reconnect
6. ✅ Clean README with architecture diagram, quickstart, and benchmarks
7. ✅ GPT-2 example works (proves framework is model-agnostic)

---

## Appendix A: Estimated Communication Times

| Model | Params | FP32 Size | Compressed (Top10%+FP16) | Transfer @4GB/s | % of 100ms step |
|-------|--------|-----------|--------------------------|-----------------|------------------|
| ResNet-50 | 25M | 100 MB | ~15 MB | 3.75 ms | 3.75% |
| GPT-2 (124M) | 124M | 496 MB | ~74 MB | 18.5 ms | 18.5% |
| ViT-Base | 86M | 344 MB | ~52 MB | 13 ms | 13% |
| GPT-2 (355M) | 355M | 1.42 GB | ~213 MB | 53 ms | 53% ⚠️ |

**Takeaway**: Models up to ~150M params have very manageable communication overhead with compression. Above 300M params, communication becomes the bottleneck even with TB4 — this is where pipeline parallelism becomes necessary.

## Appendix B: Quick Reference for Cursor

### When implementing any file, check:
1. Does it handle the MPS ↔ CPU ↔ Network data flow correctly?
2. Does it respect the Air's memory limit (~10 GB usable)?
3. Is tensor serialization using raw bytes (not pickle/protobuf)?
4. Is there proper `torch.mps.synchronize()` before timing or sending?
5. Are errors caught gracefully with fallback behavior?
6. Is the code async where it should be (network I/O)?

### Naming conventions:
- Classes: `PascalCase` (e.g., `GradientCompressor`, `ClusterCoordinator`)
- Functions: `snake_case` (e.g., `send_tensor`, `calibrate_throughput`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TOPK_RATIO`, `HEARTBEAT_INTERVAL_SEC`)
- Proto messages: `PascalCase` (e.g., `RegisterRequest`)

### Import order:
1. Standard library
2. Third-party (torch, grpc, rich)
3. Local (macfleet.*)

### Type hints:
- Use type hints on all function signatures
- Use `dataclasses` for configs
- Use `Optional[]` explicitly for nullable params
