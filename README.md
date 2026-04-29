# TumorClassifier (MEDIQA-CORE) 🧠

**Multimodal Reasoning and Reconciliation in Radiology**

A tri-modal deep learning system for brain tumor classification that jointly processes multi-sequence MRI volumes, whole-slide histopathology images (WSI), and clinical pathology reports — mirroring the real-world clinical diagnostic workflow where no single data source tells the full story.

---

## Overview

Standard single-modality classifiers discard critical diagnostic context. A radiologist reads the MRI. A pathologist reads the slide. A clinician reads the report. MEDIQA-CORE fuses all three into a single end-to-end pipeline, with a gated fusion mechanism that learns how much to trust each modality at inference time — including when one is missing.

---

## Architecture

### MRI Agent — 3D CNN
Processes stacked T1, T1c, T2, and FLAIR sequences as a joint `4x64x64x64` volumetric input through four Conv3D blocks (4 to 256 channels), capturing spatial tumor morphology across all MRI sequences simultaneously.

### WSI Agent — ResNet18 + Attention MIL
Handles gigapixel H&E whole-slide images using a pretrained ResNet18 backbone with attention-based Multiple Instance Learning (MIL) pooling. Rather than downsampling the entire slide, MIL learns which tile regions are the most diagnostically discriminative and weights them accordingly.

### Text Agent — BioBERT
Encodes free-text pathology reports through a 12-layer BioBERT encoder (768-d), capturing biomedical domain terminology including IDH mutation status, histological subtypes, Ki-67 proliferation indices, and margin assessments.

### Fusion
All three modality encoders project to a shared **256-d embedding space**, preventing any single modality from dominating fusion by vector magnitude. Pairwise cross-modal co-attention (4-head) then lets each modality attend to the others before fusion. A gated fusion layer (FC 768 to 3 + Sigmoid) learns adaptive per-modality importance weights, allowing the model to gracefully handle missing modalities at inference by suppressing zero-padded vectors.

```
MRI (3D CNN) ──────────────────────────────►
                                             Cross-Modal     Gated     Classifier
WSI (ResNet18 + Attention MIL) ────────────►  Co-Attention   Fusion ──►
                                             (4-head, pairwise)
Report (BioBERT) ──────────────────────────►
```

---

## Key Design Decisions

**Shared embedding space.** Without projecting all modalities to the same dimensionality before fusion, a high-norm modality (e.g. BioBERT's 768-d output) would dominate dot-product attention scores regardless of its actual diagnostic relevance.

**Gated fusion over simple concatenation.** Simple concatenation treats all modalities equally. The sigmoid gate learns task-specific modality importance, allowing the model to suppress unreliable or absent inputs without hard-coded rules.

**Attention MIL for WSI.** Whole-slide images at full resolution are gigapixels -- they cannot be fed directly into a CNN. Attention MIL operates on tiles, learns bag-level labels without tile-level annotation, and produces interpretable attention weights that highlight diagnostically relevant regions.

**Modality dropout during training.** Randomly zeroing one modality per batch forces the model to never over-rely on any single source, improving robustness when a modality is unavailable at deployment.

---

## Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![BioBERT](https://img.shields.io/badge/BioBERT-NLP-0e6655?style=flat-square)
![ResNet18](https://img.shields.io/badge/ResNet18-AttentionMIL-8e44ad?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

---

## Repository Structure

```
TumorClassifier/
└── TumorClassifier.ipynb   # Full pipeline: data loading, model definition,
                            # training loop, fusion, evaluation
```

---

## Future Work

- Replace synthetic pathology images with real WSI datasets (TCGA)
- Add SHAP-based modality attribution to explain which input drove each prediction
- Extend to glioma grading (Grade II / III / IV) beyond binary classification
- Explore cross-attention transformers as an alternative to the MIL pooling stage
