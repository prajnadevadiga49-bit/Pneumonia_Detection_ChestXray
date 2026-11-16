# Pneumonia Detection from Chest X-Rays: ConvNeXt-Tiny vs. EfficientNet-B0 (and their Ensemble)

**Course:** Neural Networks and Deep Learning (NNDL)
**Task:** Binary classification of paediatric chest X-rays: `NORMAL` vs. `PNEUMONIA`
**Framework:** PyTorch / torchvision
**Notebook:** `pneumonia_detection_convnext_efficientnet_ensemble.ipynb`

---

## 1. Project overview

Pneumonia is diagnosed in practice from chest radiographs, but reading them is time-consuming and error-prone.
This project trains and compares two modern, ImageNet-pretrained convolutional networks on the public
*Chest X-Ray Images (Pneumonia)* dataset, and evaluates a simple ensemble of the two:

| # | Model | Idea |
|---|---|---|
| 1 | **ConvNeXt-Tiny** | A convolutional network modernised with Vision-Transformer-style design (7x7 depthwise kernels, LayerNorm, GELU) |
| 2 | **EfficientNet-B0** | A compact, compound-scaled network (MBConv blocks with squeeze-and-excitation, about 4.2M parameters) |
| 3 | **Ensemble** | Equal-weight average of the two models' predicted pneumonia probabilities (nothing tuned on the test set) |

The pipeline (data loading, splitting, augmentation, loss, optimiser, evaluation) is identical for all models,
so differences in the results come from the architectures.

## 2. Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kermany et al.),
mirrored at https://data.mendeley.com/datasets/rscbjbr9sj/3.

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train pool (`train/`) | 1,341 | 3,875 | 5,216 |
| Test (`test/`) | 234 | 390 | 624 |

- The dataset's own `val/` folder holds only 16 images, which is far too few for reliable model selection, so it is **not used**.
- Instead, the 5,216 training images are split **85 / 15 (stratified)** into **4,433 train** and **783 validation** images.
- The official `test/` folder is kept untouched as the held-out test set (624 images).
- Pneumonia images outnumber normal images about 3 to 1, which is handled with a `WeightedRandomSampler`.

![Sample images](figures/sample_images.png)

## 3. Method

### Pipeline (shared by all models)

| Step | Setting |
|---|---|
| Input | Grayscale converted to 3 channels, resized to 224x224, ImageNet mean/std normalisation |
| Augmentation (train only) | Random horizontal flip (p=0.3), random rotation (10 degrees) |
| Split | 85/15 stratified train/validation, `random_state=42` |
| Class imbalance | `WeightedRandomSampler` (inverse class frequency) |
| Loss | Cross-entropy |
| Optimiser | Adam, `lr = 2e-4`, `ReduceLROnPlateau` (factor 0.5, patience 2) on validation loss |
| Training | 10 epochs, batch size 32; the checkpoint with the lowest validation loss is restored |
| Evaluation | Held-out test set, default decision threshold 0.5 |

### Architectures

Both models start from ImageNet weights. The early layers are **frozen** (generic edges and textures) and kept in
evaluation mode during training; the last stage(s) and a new classification head are trained.

| | ConvNeXt-Tiny | EfficientNet-B0 |
|---|---|---|
| Trainable backbone part | Last downsample + stage 4 (`features[6:]`) | Last two blocks (`features[7:]`) |
| New head | GAP, LayerNorm, Dropout(0.3), Linear(768, 128), GELU, Dropout(0.2), Linear(128, 2) | GAP, Dropout(0.3), Linear(1280, 128), ReLU, Dropout(0.2), Linear(128, 2) |
| Trainable parameters | 15,570,818 | 1,293,618 |
| Total parameters | 27,918,818 | 4,171,774 |

### Why these models?

Recent benchmark papers on this dataset report that modern pretrained backbones clearly outperform small from-scratch
CNNs. One 2026 paediatric-pneumonia architecture comparison reported ConvNeXt-Tiny at about 97.6% accuracy,
ahead of EfficientNet-B4 at about 95.1%, and several other studies place EfficientNet-B0 among the top performers
despite its small size. ConvNeXt-Tiny is therefore the main model, and EfficientNet-B0 is a lightweight,
architecturally different second model (BatchNorm and MBConv vs. LayerNorm and large kernels), which also makes it a
useful ensemble partner.

## 4. Results

All numbers are on the **held-out test set (624 images)** at the default 0.5 threshold, from a single run (seed 42).

| Model | Accuracy | Precision | Recall (PNEUMONIA) | F1 | ROC-AUC | NORMAL recall (specificity) |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.7756 | 0.7367 | 0.9974 | 0.8475 | **0.9802** | 0.41 |
| EfficientNet-B0 | **0.8285** | 0.7870 | 0.9949 | 0.8788 | 0.9732 | **0.55** |
| Ensemble | 0.8093 | 0.7673 | 0.9974 | 0.8673 | 0.9793 | 0.50 |

Confusion matrices (rows = actual, columns = predicted):

| Model | NORMAL correct | NORMAL flagged as pneumonia | PNEUMONIA missed | PNEUMONIA found |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny | 95 | 139 | 1 | 389 |
| EfficientNet-B0 | 129 | 105 | 2 | 388 |
| Ensemble | 116 | 118 | 1 | 389 |

![Training curves](figures/training_curves.png)
![Confusion matrices](figures/confusion_matrices.png)
![ROC curves](figures/roc_curves.png)

Validation accuracy reached about 99% for both models by the final epochs (ConvNeXt-Tiny best validation loss 0.0300 at epoch 9;
EfficientNet-B0 best validation loss 0.0297 at epoch 8). Each epoch took roughly 75 seconds on the GPU used.

### Comparison with the baseline notebook

The reference notebook this project follows (a small CNN trained from scratch, and a frozen ResNet18) reported:

| Model | Test accuracy | ROC-AUC | NORMAL recall | PNEUMONIA recall |
|---|---:|---:|---:|---:|
| SimpleCNN (scratch) | 0.8381 | 0.9285 | 0.59 | 0.98 |
| ResNet18 (frozen backbone) | 0.9022 | 0.9534 | 0.81 | 0.96 |

> Their saved run counted every image twice (10,432 training / 1,248 test images instead of 5,216 / 624) because of
> duplicated nested folders left by the extraction step, so its validation scores are inflated. Duplicating the test set
> does not change accuracy or recall proportions, so the test figures above are still comparable. This notebook avoids the problem by loading only
> images directly inside each class folder and by asserting that there are no duplicate files.

## 5. Analysis

1. **Ranking quality is excellent; the default threshold is not.** ROC-AUC is 0.97 to 0.98 for all three systems, higher
   than the baseline's 0.93 to 0.95. But at the default 0.5 threshold, the models flag many healthy images as pneumonia
   (NORMAL recall only 41% to 55%). On accuracy and NORMAL recall at 0.5, the baseline's frozen ResNet18 (0.90 and 0.81)
   was **better** than our models. We do not claim that ours beat the baseline overall: ours is better at ranking cases and at catching pneumonia
   (99.5% to 99.7% recall vs. 95.6%), the baseline is better at not over-calling pneumonia at the default threshold.
2. **Validation vs. test gap.** Validation accuracy of about 99% versus test accuracy of 78% to 83% is a large drop.
   Two reasons are likely: (a) the official test split is known to look different from the training pool (a dataset shift), and
   (b) the validation split is by image, not by patient, so several X-rays of the same child can land in both train and validation, making validation optimistic.
3. **Backbone comparison.** EfficientNet-B0 has the best accuracy and specificity at 0.5; ConvNeXt-Tiny has the best ROC-AUC and the
   fewest missed pneumonia cases (1 vs. 2). With only 624 test images (234 normal), these differences are small and may be within run-to-run noise,
   and EfficientNet-B0 does this with about 12 times fewer trainable parameters.
4. **Ensemble.** Averaging landed between the two members on accuracy and NORMAL recall, and only tied the best member on PNEUMONIA recall (ConvNeXt-Tiny's 0.9974); it did not beat the best member on any other headline metric in this run.
5. **Clinical view.** For screening, missed pneumonia (false negatives) is the costly error, and all three systems miss only 1 or 2 of 390 cases.
   The price is a high false-alarm rate on healthy patients, so these models are not ready for use as a standalone diagnostic tool.

## 6. How to run

### Requirements

- Python 3.9+
- `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `jupyter`
- A CUDA GPU is strongly recommended (about 75 s per epoch on a GPU; the CPU will be far slower)
- Internet connection on the first run (PyTorch downloads the ImageNet weights, about 130 MB in total)

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn jupyter
```

### Steps

1. Download the dataset zip from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   (`archive.zip`, about 1.2 GB).
2. Put the zip in the **same folder as the notebook**.
3. Open `pneumonia_detection_convnext_efficientnet_ensemble.ipynb` and run all cells in order.
   The first cells unzip the data into `data/` and locate `train/` and `test/` automatically.
4. Check that the sanity-check cell prints `Loaded 5216 train+val images and 624 test images`.

> **Quick smoke test:** set `EPOCHS = 1` in the "Shared training / evaluation utilities" cell, run once end to end,
> then set it back to `10` for the real run.
>
> **GPU out of memory?** Lower `BATCH_SIZE = 32` to `16`.

### Project structure

```
.
├── pneumonia_detection_convnext_efficientnet_ensemble.ipynb   # full pipeline
├── README.md
├── figures/                                                   # plots from the run
│   ├── sample_images.png
│   ├── training_curves.png
│   ├── confusion_matrices.png
│   └── roc_curves.png
├── archive.zip                                                # dataset (downloaded by you)
└── data/                                                      # created by the notebook
```

## 7. Limitations

- Single hospital, paediatric patients (ages 1 to 5): results do not automatically carry over to adults or other scanners.
- The train/validation split is by image, not by patient, so validation scores are optimistic (see Analysis).
- One run with one random seed; with a 624-image test set, small differences between models are not statistically reliable.
- The 0.5 decision threshold is not tuned, and the models are not calibrated; no external dataset was used.
- This is a coursework exercise, **not a medical device**, and must not be used for clinical decisions.

## 8. Possible improvements

- Tune the decision threshold (and calibrate probabilities with temperature scaling) on the validation set to trade some pneumonia
  sensitivity for much better NORMAL recall; the high ROC-AUC suggests there is a lot of room.
- Split train and validation by patient (patient IDs can be read from the file names).
- Contrast normalisation (e.g. CLAHE) and stronger regularisation to reduce the validation-to-test gap.
- Repeat with several seeds and report confidence intervals; evaluate on an external chest X-ray dataset.
- Add Grad-CAM heatmaps to check that the models look at the lung fields.

## 9. References

- Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, 2018 (dataset).
- Liu et al., *A ConvNet for the 2020s* (ConvNeXt), CVPR 2022.
- Tan and Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
- Recent benchmark studies of deep-learning architectures for paediatric pneumonia on this dataset, which guided the model choice
  (e.g. an architecture-comparison study with ensembling and external evaluation, Applied Sciences, 2026).
