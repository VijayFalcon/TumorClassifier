#MEDIQA-CORE 🧠
Multimodal Reasoning & Reconciliation in Radiology
A tri-modal deep learning system for brain tumor classification that jointly processes multi-sequence MRI volumes, whole-slide histopathology images (WSI), and clinical pathology reports to mirror the real-world clinical diagnostic workflow.
Architecture
MRI Agent — 3D CNN backbone (4 Conv3D blocks, 4→256 channels) processes stacked T1, T1c, T2, and FLAIR sequences as a joint 4×64×64×64 volumetric input, capturing spatial tumor morphology across all sequences simultaneously.
WSI Agent — Pretrained ResNet18 backbone with attention-based Multiple Instance Learning (MIL) pooling handles gigapixel H&E slides by learning which tile regions are most diagnostically discriminative.
Text Agent — BioBERT encoder (12-layer, 768-d) encodes free-text pathology reports including domain-specific terminology like IDH status, histological subtypes, and Ki-67 indices.
Key Design Choices

All three modality encoders project to a shared 256-d embedding space to prevent any single modality from dominating fusion by vector magnitude
Cross-modal co-attention (4-head, pairwise across all three modalities) enables each modality to attend to the others before fusion
Gated fusion mechanism (FC 768→3 + Sigmoid) learns adaptive importance weights per modality, allowing the model to suppress zero-padded vectors when modalities are unavailable at inference time

Stack
PyTorch HuggingFace Transformers BioBERT ResNet18 3D CNN Attention MIL
