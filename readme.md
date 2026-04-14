# DLOps Assignment-5

This submission contains:
- `q1_vit_lora_cifar100.py` for LoRA-based ViT-S finetuning on CIFAR-100
- `q2_art_attacks_and_detection.py` for IBM ART adversarial attacks and detection on CIFAR-10
- `report.pdf` / `report.tex` for the final report
- `Dockerfile.q1` and `Dockerfile.q2` for containerized execution

## Links
- Hugging Face model: [subilal/lora-vit-cifar100](https://huggingface.co/subilal/lora-vit-cifar100)
- WandB Q1: [vit-lora-cifar100](https://wandb.ai/abhikashyap4563-iit-jodhpur/vit-lora-cifar100)
- WandB Q2: [dlops-assignment5-q2-art](https://wandb.ai/abhikashyap4563-iit-jodhpur/dlops-assignment5-q2-art)

## Q1 Summary
Best fixed-grid LoRA configuration:
- `rank = 4`
- `alpha = 8`
- `dropout = 0.1`
- test accuracy = `0.8778`

### Q1 Overall Test Results

| LoRA | Rank | Alpha | Dropout | Test Accuracy | Trainable Params |
|---|---:|---:|---:|---:|---:|
| No | 0 | 0 | 0.0 | 0.7791 | 38500 |
| Yes | 2 | 2 | 0.1 | 0.8734 | 75364 |
| Yes | 2 | 4 | 0.1 | 0.8772 | 75364 |
| Yes | 2 | 8 | 0.1 | 0.8746 | 75364 |
| Yes | 4 | 2 | 0.1 | 0.8727 | 112228 |
| Yes | 4 | 4 | 0.1 | 0.8764 | 112228 |
| Yes | 4 | 8 | 0.1 | **0.8778** | 112228 |
| Yes | 8 | 2 | 0.1 | 0.8725 | 185956 |
| Yes | 8 | 4 | 0.1 | 0.8751 | 185956 |
| Yes | 8 | 8 | 0.1 | 0.8775 | 185956 |
| Yes | 8 | 8 | 0.1 | 0.8712 | 185956 |
| Yes | 2 | 2 | 0.1 | 0.8510 | 75364 |

## Q2 Summary
Latest recorded Q2 run:
- clean classifier test accuracy = `0.8350`
- best PGD detector accuracy = `0.7147`
- best BIM detector accuracy = `0.7215`

### Q2 FGSM Comparison

| Epsilon | Clean Acc | Scratch Acc | ART Acc | Scratch Drop | ART Drop |
|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.8350 | 0.8350 | 0.8350 | 0.0000 | 0.0000 |
| 0.0100 | 0.8350 | 0.6971 | 0.7646 | 0.1379 | 0.0704 |
| 0.0200 | 0.8350 | 0.5568 | 0.6449 | 0.2782 | 0.1901 |
| 0.0300 | 0.8350 | 0.4302 | 0.5226 | 0.4048 | 0.3124 |
| 0.0500 | 0.8350 | 0.2517 | 0.3474 | 0.5833 | 0.4876 |

### Q2 Detector Results

| Detector Attack | Best Detection Accuracy |
|---|---:|
| PGD | 0.7147 |
| BIM | **0.7215** |

## Docker

### Q1
```bash
docker build -f Dockerfile.q1 -t assignment5-q1 .
docker run --rm -it --env-file .env assignment5-q1
```

### Q2
```bash
docker build -f Dockerfile.q2 -t assignment5-q2 .
docker run --rm -it --env-file .env assignment5-q2
```

## Notes
- The detailed discussion, figures, and observations are included in `report.pdf`.
- The scripts support reading `HF_TOKEN` and `WANDB_API_KEY` from `.env`.
