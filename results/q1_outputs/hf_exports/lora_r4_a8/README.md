# lora_r4_a8

- Base model: `vit_small_patch16_224`
- Dataset: `CIFAR-100`
- Test accuracy: `0.8778`
- LoRA enabled: `Yes`
- LoRA rank: `4`
- LoRA alpha: `8`
- Trainable params: `112228`

Attention note: timm ViT exposes a fused `qkv` projection, so LoRA is attached to the combined QKV layer rather than separate `q_proj`, `k_proj`, and `v_proj` modules.

This folder was exported automatically from the assignment training script.
