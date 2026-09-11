# Zero-Configuration Endoscopic Visual Odometry with Tissue Motion Decoupling

A monocular endoscopic visual odometry (VO) system that estimates **metric-scale (millimeter) camera poses and tissue motion trajectories from a raw RGB frame sequence alone** — without external calibration targets or scale references.

Built on a five-stage pipeline: depth estimation → feature matching → static-point selection → PnP pose solving → scale correction, followed by camera-motion/tissue-motion decoupling.

## Key Features

- **Zero configuration**: only RGB images are required; the absolute scale is recovered automatically through a PnP-based depth-scale correction scheme (`pnp_scale`).
- **Millimeter-scale depth**: a 64-bin classification + residual hybrid head outputs depth in the 1–500 mm range.
- **Static/moving point separation**: adaptive geometric filtering (median + 1.0·std) with a trained motion classifier for robust VO under tissue deformation.
- **Tissue motion decoupling**: chained inter-frame tracking + world-coordinate back-projection quantifies tissue motion in absolute 3D.
- **One-command inference**: `run_pipeline.py` outputs a 3D trajectory plot, per-frame depth visualizations, and the absolute camera pose sequence.

## Pipeline Overview

1. **Depth backbone** — Monodepth2-style ResNet-18 encoder + depth decoder with a 64-bin classification head (1–500 mm, log-uniform bins), trained self-supervised with scale-anchoring and variance lower-bound losses.
2. **Feature matching** — LoFTR (kornia) matches + Lucas-Kanade optical-flow tracking to establish inter-frame correspondences.
3. **Static-point selection** — residual-flow based adaptive thresholding picks geometrically consistent points.
4. **Pose solving** — EPnP estimates the relative pose; a two-level scale correction (global scale back-inference + `pnp_scale` refinement) restores absolute scale.
5. **Scale correction & decoupling** — inter-frame depth-consistency correction suppresses scale drift; subtracting camera-induced apparent motion yields tissue motion, quantified in absolute 3D.

## Key Results

Depth/VO comparison on the C3VDv2 colonoscopy benchmark (unified backend, depth method replaced):

| Method | ATE (mm) ↓ | RPE-R (°) ↓ | Scale Error ↓ | Success |
|--------|-----------|------------|---------------|---------|
| Monodepth2 | 7.12 | 0.35 | 55 % | — |
| ManyDepth  | —     | 2.06 | —             | — |
| Lite-Mono  | —     | —     | 198 %         | — |
| **Ours**   | **5.08** | **0.43** | **25 %** | **100 %** |

> Baseline absolute values above are indicative; please cross-check against Table 1 of the paper before publication.

Camera-pose comparison on public benchmarks (5-frame-window ATE, mean ± std, mm):

| Method | SCARED | EndoSLAM |
|--------|--------|----------|
| AF-SfMLearner | **46.65±27.52** | 2.66±1.52 |
| Endo-FASt3r   | 46.68±27.54 | 2.39±1.46 |
| BodySLAM      | 48.63±29.11 | 0.80±0.48 |
| **Ours**      | 49.28±29.36 | **0.64±0.18** |

## Repository Structure

```
├── run_pipeline.py              # one-command VO inference entry
├── test_v6_dyendovo.py          # integration test / evaluation
├── train.py                     # self-supervised depth training (single source)
├── train_depth_flow.py          # depth + optical-flow training
├── train_motion_classifier.py   # motion classifier training
├── train_multi_source.py        # multi-source training
├── trainer.py                   # training loop
├── options.py / layers.py / utils.py / kitti_utils.py
├── dyendovo_network.py          # depth/pose/motion network definitions
├── dyendovo_dataset.py          # dataset helpers & GT parsing
├── motion_classifier.py         # motion classification model
├── networks/                    # encoder/decoder/PoseCNN/MotionEncoder
├── v6_pipeline/                 # VO pipeline (PnP, BA, track manager, filters)
├── datasets/                    # C3VD / EndoSLAM / SCARED / KITTI loaders
├── config/                      # pipeline config (v6_config.yaml)
├── splits/                      # dataset split files
├── scripts/                     # utility scripts
├── zhong/                       # paper experiment scripts
├── baselines/                   # self-written baseline evaluation scripts
└── dyendovo_output/             # model weights (Git LFS)
```

## Installation

Requirements:

- Python 3.11
- PyTorch ≥ 2.0 (tested with 2.8.0 + CUDA 12.8)
- numpy, opencv-python, torchvision, pillow, matplotlib, scipy, tqdm
- kornia (LoFTR feature matching)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy opencv-python pillow matplotlib scipy tqdm kornia
```

## Model Weights

Pre-trained weights are tracked with Git LFS under `dyendovo_output/`:

- `final_model.pth` — final depth/VO model
- `best_model.pth` — best checkpoint

Clone with LFS:

```bash
git lfs install
git clone <repo-url>
```

## Quick Start (Inference)

Run the VO pipeline on a folder of RGB frames:

```bash
python run_pipeline.py \
  --input  <path/to/rgb_frames> \
  --output <path/to/results> \
  --fx 500 --fy 500 --cx 640 --cy 360 \
  --model dyendovo_output/final_model.pth
```

Outputs:

- `output_trajectory_3d.png` — 3D camera trajectory
- `depth_viz/` — per-frame depth visualization
- `output_abs_poses.txt` — absolute camera poses (4×4 matrices, row-major)

## Training

Self-supervised depth training (C3VD `*_color.png` or EndoSLAM `generated/rgb_warped` layouts are auto-detected):

```bash
python train.py --data_path <path/to/sequence> --log_dir ./logs/c3vd --epochs 50
```

Depth + flow, motion classifier, and multi-source variants:

```bash
python train_depth_flow.py        --data_path <path> --log_dir ./logs/flow
python train_motion_classifier.py --data_path <path> --log_dir ./logs/motion
python train_multi_source.py      --data_path <path> --log_dir ./logs/multi
```

## Evaluation

Pose evaluation against ground truth:

```bash
python test_v6_dyendovo.py --seq <seq_name> --data_root <path/to/data> --mode motionnet
```

Five-frame-window ATE (used in the paper) for baseline comparisons:

```bash
python baselines/compute_5frame_ate.py --rel_npz <pred_poses.npz> --gt_npz <gt_poses.npz> --window 5
```

## License

This project builds upon [Monodepth2](https://github.com/nianticlabs/monodepth2) and inherits its **non-commercial license**. See the Monodepth2 license terms for details. Please also respect the licenses of the datasets (C3VD, EndoSLAM, SCARED) and third-party baselines used in the comparisons.

## Acknowledgements

We thank the authors of Monodepth2, LoFTR/kornia, C3VD, EndoSLAM, SCARED, and the compared baselines (BodySLAM, Endo-FASt3r, AF-SfMLearner) for releasing their code and data.
