# Final paired artifacts

Campaign: `codex-6h-final-1h-20260719-101826`  
Host: `lambda-quad`  
Code: `f286f49ab912a1be03fd1ccae51ed2e4eba9f2e9`  
Remote directory: `/home/mark/worktrees/autoresearch-campaign/checkpoints/codex-6h-final`

Both checkpoints were loaded on CPU after the campaign. Each contains 52 model
tensors, the exact model config and hyperparameters, BF16 metadata, and a
3,600-second measured training budget. They are intentionally not committed to
Git because each file exceeds GitHub's ordinary 100 MB file limit.

| GPU / seed | Validation BPB | Counted steps | Tokens | Peak VRAM | Checkpoint | Bytes | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0 / 101 | 1.282301 | 1,300 | 681.574M | 3,182.83 MB | `codex-6h-final-1h-20260719-101826-r01-gpu0-final_bf16_lr_five_sixteenths-s101-val1.282301.pt` | 125,859,931 | `303fc01c43f8f0693d67d389bd568b544e58eac1370ebf339f767546d4b79966` |
| 1 / 202 | 1.280723 | 1,381 | 724.042M | 3,182.83 MB | `codex-6h-final-1h-20260719-101826-r01-gpu1-final_bf16_lr_five_sixteenths-s202-val1.280723.pt` | 125,859,931 | `deee51a8b98aacfadb94b4b7973a01c27f883c0443bce7a72530d79f9d8519f6` |

Paired mean: `1.281512` BPB. Spread: `0.001578` BPB. Both fixed-probe
trajectories finished at their best values with zero regression events. The
checked-in sample copies are in `campaigns/final-samples/`; their originals sit
beside the checkpoints on the GPU host.
