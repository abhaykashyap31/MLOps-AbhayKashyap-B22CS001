# LoRA ViT CIFAR-100 Experiment Results

| name                | lora   |   rank |   alpha |   dropout |   test_acc |   params | hub_pushable   |
|:--------------------|:-------|-------:|--------:|----------:|-----------:|---------:|:---------------|
| baseline            | No     |      0 |       0 |       0   |     0.7791 |    38500 | False          |
| lora_r2_a2          | Yes    |      2 |       2 |       0.1 |     0.8734 |    75364 | True           |
| lora_r2_a4          | Yes    |      2 |       4 |       0.1 |     0.8772 |    75364 | True           |
| lora_r2_a8          | Yes    |      2 |       8 |       0.1 |     0.8746 |    75364 | True           |
| lora_r4_a2          | Yes    |      4 |       2 |       0.1 |     0.8727 |   112228 | True           |
| lora_r4_a4          | Yes    |      4 |       4 |       0.1 |     0.8764 |   112228 | True           |
| lora_r4_a8          | Yes    |      4 |       8 |       0.1 |     0.8778 |   112228 | True           |
| lora_r8_a2          | Yes    |      8 |       2 |       0.1 |     0.8725 |   185956 | True           |
| lora_r8_a4          | Yes    |      8 |       4 |       0.1 |     0.8751 |   185956 | True           |
| lora_r8_a8          | Yes    |      8 |       8 |       0.1 |     0.8775 |   185956 | True           |
| lora_partial_freeze | Yes    |      8 |       8 |       0.1 |     0.8712 |   185956 | False          |
| optuna_best_r2_a2   | Yes    |      2 |       2 |       0.1 |     0.851  |    75364 | True           |