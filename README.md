# PanimmuneNet

PanimmuneNet is a PyTorch framework for learning immune receptor–ligand
interactions from amino-acid sequences. The repository currently supports two
related tasks:

1. **pMHC prediction** — regression of peptide–MHC-I binding scores.
2. **TCR–pMHC prediction** — binary classification of T-cell
   receptor–peptide–MHC interactions.

The models use projected one-hot residue encodings rather than pretrained
protein language-model embeddings. Residue features are converted into pairwise
grids, processed by U-Net-style convolutional blocks and axial self-attention,
and passed to a convolutional prediction head.

## Model overview

### pMHC model

The pMHC model receives an MHC-I pseudosequence and a peptide:

```text
MHC pseudosequence + peptide
        ↓
projected one-hot residue and chain encodings
        ↓
pairwise representation
        ↓
U-Net + axial self-attention
        ↓
2D convolutional prediction head
        ↓
normalized binding score
```

The default configuration uses:

- MHC pseudosequence length: 34 residues
- Maximum peptide length: 15 residues
- Pair representation dimension: 32
- Sigmoid regression output in the range `[0, 1]`

Peptides shorter than 15 residues are right-padded with `X`. Peptides longer
than 15 residues are not accepted by the training data loader. MHC
pseudosequences are truncated or right-padded to 34 residues.

### TCR–pMHC model

The full model combines TCR and pMHC pairwise representations:

```text
TCRα + optional TCRβ          MHC pseudosequence + peptide
          ↓                               ↓
     TCR pair grid                    pMHC pair grid
          └──────────────┬────────────────┘
                         ↓
              full interaction grid
                         ↓
              U-Net + axial refinement
                         ↓
             binary interaction logit
```

The default TCR limits are 22 residues for TCRα and 26 residues for TCRβ. The
default combined grid contains 97 residues: 48 TCR residues, 34 MHC residues,
and 15 peptide residues. The TCR classifier emits a logit; inference converts
it to a probability with the sigmoid function.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/model_config.py` | Default pMHC, TCR, full-grid, and head configurations |
| `src/MHCpeptideEmbedding.py` | pMHC pairwise embedder |
| `src/tcrMHCpeptideEmbedding.py` | TCR and full TCR–pMHC pairwise embedders |
| `MHCpeptideEmbeddingClassifier.py` | pMHC regression model |
| `TCRmhcEmbeddingClassifier.py` | TCR–pMHC classification model |
| `PanTCR_dataload.py` | Sequence cleaning, datasets, and batch collation |
| `training_classification.py` | pMHC distributed training workflow |
| `tcr_training_classification.py` | TCR–pMHC distributed training workflow |
| `example_usage.py` | Standalone inference example for trained checkpoints |

Large datasets, model parameters, benchmark outputs, and analysis artifacts are
excluded from Git through `.gitignore`.

## Environment

The development environment is a Conda environment named `PanimmuneNet` with
the following core versions:

| Component | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.5.1 |
| CUDA runtime used by PyTorch | 12.1 |
| cuDNN | 9.1 |
| torchvision | 0.20.1 |
| torchaudio | 2.5.1 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.2 |
| Transformers | 4.57.0 |
| Datasets | 4.0.0 |
| einops | 0.8.1 |

To create a compact environment for the model and inference code:

```bash
conda create -n PanimmuneNet python=3.10 -y
conda activate PanimmuneNet
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 \
  pytorch-cuda=12.1 pandas=2.3.3 numpy=2.2.6 scipy=1.15.2 \
  -c pytorch -c nvidia -c conda-forge
```

The complete development environment additionally contains the scientific and
analysis packages used by the benchmarking, figure, and cohort-analysis
scripts. Those scripts may require packages such as Matplotlib, seaborn,
scikit-learn, Biopython, lifelines, PyDESeq2, and GSEApy.

## Usage

Run commands from the repository root. Checkpoints and data are not stored in
Git, so supply a local checkpoint path.

### pMHC binding score

```bash
python example_usage.py pmhc \
  --checkpoint model_parameter/ckpt_epoch160.pt \
  --mhc "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" \
  --peptide "SIINFEKL"
```

The command prints a normalized pMHC binding score. The sequence above is only
an input-format example; replace it with a real 34-residue MHC pseudosequence.

### TCR–pMHC binding probability

```bash
python example_usage.py tcr \
  --checkpoint model_parameter_vdjdb_piste_unipep/ckpt_epoch90.pt \
  --mhc "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" \
  --peptide "SIINFEKL" \
  --tcra "CAVRDSNYQLIW" \
  --tcrb "CASSLGQGAEAFF"
```

`--tcrb` may be omitted for an alpha-only example. The script cleans sequences
to the 20 canonical amino acids, applies the same default length rules as the
training pipeline, selects CUDA automatically when available, and reports the
device used.

Use `--device cpu`, `--device cuda`, or a specific device such as
`--device cuda:0` to override automatic device selection:

```bash
python example_usage.py --device cuda:0 pmhc \
  --checkpoint /path/to/checkpoint.pt \
  --mhc "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" \
  --peptide "SIINFEKL"
```

To see every option:

```bash
python example_usage.py --help
python example_usage.py pmhc --help
python example_usage.py tcr --help
```

## Input data formats

The pMHC training data loader expects a CSV file with:

| Column | Description |
| --- | --- |
| `Antigen` | Peptide amino-acid sequence |
| `MHC_sequence` | MHC-I pseudosequence |
| `Label` | Numeric regression target, normally normalized to `[0, 1]` |

The integrated TCR–pMHC data loader expects:

| Column | Description |
| --- | --- |
| `Antigen` | Peptide amino-acid sequence |
| `MHC_sequence` | MHC-I pseudosequence |
| `TCR_alpha` | TCR α-chain sequence |
| `TCR_beta` | TCR β-chain sequence |
| `Label` | Binary interaction label (`0` or `1`) |

Training paths and hyperparameters are defined near the beginning of
`training_classification.py` and `tcr_training_classification.py`. The training
workflows support PyTorch DistributedDataParallel when launched in a
distributed environment.

## Stampede3 computing platform

PanimmuneNet was developed and run on a TACC Stampede3 H100 GPU node. The
following specification was verified against the
[official Stampede3 user guide](https://docs.tacc.utexas.edu/hpc/stampede3/):

| Resource | Specification |
| --- | --- |
| GPUs | 4 × NVIDIA H100 SXM5 |
| GPU memory | 96 GB per GPU |
| CPU | Intel Xeon Platinum 8468 (Sapphire Rapids) |
| CPU sockets | 2 |
| Cores | 48 per socket; 96 per node |
| Hardware threads | 1 per core; 96 per node |
| Clock rate | 2.10 GHz |
| System memory | 1 TB DDR5 |
| Cache | 80 KB L1 per core; 2 MB L2 per core; 105 MB per socket |
| Local storage | 3.5 TB `/tmp` partition |
| Interconnect | Mellanox InfiniBand NDR, split-port 200 Gb/s, direct-GPU TACC fabric |

For a four-GPU allocation, PyTorch distributed training can be launched with:

```bash
torchrun --standalone --nproc_per_node=4 training_classification.py
```

or:

```bash
torchrun --standalone --nproc_per_node=4 tcr_training_classification.py
```

Cluster allocation, launcher, and scheduler options should follow the current
TACC Stampede3 documentation and the resources assigned to the job.

## Checkpoints

The inference example recognizes both checkpoint formats produced by the
training code:

- Full checkpoints containing a `model` state dictionary and optimizer state.
- Weights-only files containing a `state_dict`.

When a full checkpoint includes saved model configuration, the inference
example reconstructs the corresponding architecture from that metadata.
Weights-only checkpoints use the defaults in `src/model_config.py`; therefore,
their architecture must match the current default configuration.

## Research status

This repository contains active research code. Model scores should be validated
for the intended dataset and should not be interpreted as clinical
recommendations.
