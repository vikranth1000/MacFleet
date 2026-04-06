# MacFleet

**Pool Apple Silicon Macs into a distributed ML training cluster.**

Zero-config discovery. N-node scaling. WiFi, Ethernet, or Thunderbolt.

```
  macfleet join                macfleet join               macfleet join
 ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
 │  MacBook Pro  │◄────────►│  MacBook Air  │◄────────►│  Mac Studio   │
 │  M4 Pro       │  WiFi /  │  M4           │  WiFi /  │  M4 Ultra     │
 │  16 GPU cores │  ETH /   │  10 GPU cores │  ETH /   │  60 GPU cores │
 │  48 GB RAM    │  TB4     │  16 GB RAM    │  TB4     │  192 GB RAM   │
 │  weight: 0.35 │           │  weight: 0.15 │           │  weight: 0.50 │
 └──────────────┘           └──────────────┘           └──────────────┘
         ▲                          ▲                          ▲
         └──────────────────────────┴──────────────────────────┘
                        Ring AllReduce (gradient sync)
```

## Features

- **Zero-Config Pooling**: `pip install macfleet && macfleet join` — auto-discovers peers via mDNS/Bonjour
- **N-Node Scaling**: Ring AllReduce for 2+ nodes (not limited to pairs)
- **Any Network**: WiFi, Ethernet, and Thunderbolt with adaptive buffer tuning
- **Dual Engine**: PyTorch+MPS and Apple MLX — pluggable via Engine protocol
- **Heterogeneous Scheduling**: Weighted batch allocation based on GPU cores + thermal state
- **Gossip Heartbeat**: Peer-to-peer failure detection, automatic coordinator election
- **Adaptive Compression**: Bandwidth-aware TopK+FP16 (auto-selects by link type: WiFi=200x, Ethernet=20x, TB4=off)
- **Framework-Agnostic Core**: Communication layer uses numpy — never imports torch/mlx
- **Health Monitoring**: Thermal, memory, battery, loss trend — composite health score per node
- **Rich TUI Dashboard**: Real-time cluster topology, training progress, and warnings

## Quick Start

```bash
pip install macfleet
```

### Join the pool

```bash
# On each Mac:
macfleet join
```

### Train a model (Python SDK)

```python
import macfleet

# PyTorch
with macfleet.Pool() as pool:
    pool.train(
        model=MyModel(),
        dataset=my_dataset,
        epochs=10,
        batch_size=128,
    )

# MLX (Apple native)
with macfleet.Pool(engine="mlx") as pool:
    pool.train(
        model=mlx_model,
        dataset=(X, y),
        epochs=10,
        loss_fn=my_loss_fn,
    )

# One-liner
macfleet.train(model=MyModel(), dataset=ds, epochs=10)

# Decorator
@macfleet.distributed(engine="torch")
def my_training():
    ...
```

### CLI commands

```bash
macfleet info       # Local hardware profile
macfleet status     # Discover pool members on the network
macfleet diagnose   # System health check (MPS, thermal, network)
macfleet train      # Demo training on synthetic data
macfleet bench      # Benchmark compute, network, and allreduce
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI: macfleet join | status | train | bench | info | diagnose  │
│  SDK: macfleet.Pool() | macfleet.train()                        │
├─────────────────────────────────────────────────────────────────┤
│  Training: DataParallel | TrainingLoop | WeightedSampler        │
├─────────────────────────────────────────────────────────────────┤
│  Engines: TorchEngine (PyTorch+MPS) | MLXEngine (Apple MLX)    │
├─────────────────────────────────────────────────────────────────┤
│  Compression: TopK + FP16 + Adaptive pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│  Pool: Agent | Registry | Discovery | Scheduler | Heartbeat     │
├─────────────────────────────────────────────────────────────────┤
│  Communication: PeerTransport | WireProtocol | Collectives      │
├─────────────────────────────────────────────────────────────────┤
│  Monitoring: Thermal | Health | Throughput | Dashboard            │
└─────────────────────────────────────────────────────────────────┘
```

## Development

```bash
git clone https://github.com/yourusername/MacFleet.git
cd MacFleet
pip install -e ".[dev]"
make test          # 268 tests
make bench         # compute + network + allreduce benchmarks
make lint          # ruff + mypy
```

## Requirements

- Python 3.11+
- macOS with Apple Silicon (M1/M2/M3/M4)
- PyTorch 2.1+ (for torch engine)
- MLX 0.5+ (optional, for mlx engine)

## License

MIT
