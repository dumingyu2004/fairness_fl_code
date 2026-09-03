# Baseline Matrix Progress — 2026-09-04

## Queue
- 526 jobs queued (`tool/experiment_queue/jobs_20260904.tsv`), scheduler: `tool/experiment_queue/scheduler.sh` (tmux `fl_queue` on Ronnie, concurrency 4, MPS enabled, resume-on-restart)
- Order: FedAvg remainder (22) -> PraFFL (24) -> FedFACT (24) -> generic FL baselines FedProx/Scaffold/FederatedNova/FedRep/FedProto/LoGoFair (144) -> PDFFed (24) -> 12 new algorithms (288)
- Common config: full data (system_data_count=2e6), 50 rounds, 3 repeats (base_seed 42), micro-batch 32 (effective 256 via accumulation), test batch 512, AMP, resume, shared partition cache

## Completed
- FedAvg bios Dirichlet05 20Clients (full data, 3 repeats): ACC 0.6747±0.1258, DEO 0.0119±0.0120, FR 0.9881±0.0120, HM 0.7973±0.0859, SPD 0.0261±0.0399
- FedAvg moji Dirichlet05 20Clients (2000-sample pipeline validation, 3 repeats): ACC 0.6985±0.0009

Note: high repeat variance on bios full data (ACC 0.575-0.816) under Dirichlet05 label skew + 0.1 client fraction.

## In progress (as of 2026-09-04 07:45 CST)
- FedAvg moji Dirichlet05 20Clients (repeat_00, round ~33/50)
- FedAvg moji Dirichlet01 20/30/40Clients (repeat_00, rounds ~30-36/50)

## Measured throughput (important)
- Aggregate GPU throughput is constant under concurrency (saturated); concurrency spreads progress, does not multiply it.
- One moji config ≈ 20-24 GPU-hours; per-round full-test eval (448k rows) is ~40-50% of runtime.
- Full queue at current settings is months of compute. Pending decisions: eval subsampling (~40% saving), rounds 50->25-30, moji train subsampling (~4x saving).

## Ops notes
- Batch runner GPU_POOL enabled: ["0"] (was [], CPU-only)
- Worktrees need bert-base-uncased symlink for offline HF loading
- Scheduler counts CUDA contexts via nvidia-smi (dataloader workers must not be counted); MPS server process excluded
