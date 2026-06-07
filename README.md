# Generative Vicinity Distillation (GVD)

This repository provides the official implementation of Generative Vicinity Distillation (GVD).

## Training

The training script follows the format:

```bash
bash tools/dist_train.sh CONFIG MODEL EXP_ROOT BASE_EXP --kd METHOD
```

### Arguments

* **CONFIG**: Training configuration file.
* **MODEL**: Model to train.
* **EXP_ROOT**: Directory where experiment outputs and checkpoints are saved.
* **BASE_EXP**: Experiment name used to identify checkpoints and results.
* **METHOD** (`--kd`): Knowledge distillation method.

---

## 1. Train Teacher

```bash
bash tools/dist_train.sh \
    configs/train_cifar100_ResNet50_to_MobileNetV2.yaml \
    cifar_ResNet50 \
    exp_im_teachers \
    cifar100_ResNet50 \
    --kd ''
```

---

## 2. Train Latent Diffusion Prior

```bash
bash tools/dist_train.sh \
    configs/train_gen_prior.yaml \
    latentdiffusion \
    exp_im_teachers \
    cifar100_ResNet50_LD \
    --kd ''
```

---

## 3. Train Student

### MSE Baseline

```bash
bash tools/dist_train.sh \
    configs/train_cifar100_ResNet50_to_MobileNetV2.yaml \
    cifar_MobileNetV2 \
    exp_cifar100_disparate/cifar_MobileNetV2 \
    mse \
    --kd 'mse'
```

### GVD

```bash
bash tools/dist_train.sh \
    configs/train_cifar100_ResNet50_to_MobileNetV2.yaml \
    cifar_MobileNetV2 \
    exp_cifar100_disparate/cifar_MobileNetV2 \
    GVD \
    --kd 'GVD'
```
