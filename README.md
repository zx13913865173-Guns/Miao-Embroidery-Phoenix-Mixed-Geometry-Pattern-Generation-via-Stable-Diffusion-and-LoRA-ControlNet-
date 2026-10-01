# Miao Embroidery Phoenix Mixed-Geometry Pattern Generation

Official code for the paper:

> **Miao Embroidery Phoenix Mixed-Geometry Pattern Generation via Stable Diffusion and LoRA-ControlNet: Line–Curve–Color Separate Evaluation and Spatial-Partition Fusion**

## Overview

This repository implements a **spatial-partition fusion strategy** with **dual ControlNet units** for generating Miao embroidery phoenix patterns. It is **not** a per-pixel weighted average approach.

- **Outer-frame region**: controlled by **MLSD** with condition strength **0.7**
- **Interior region**: controlled by **Lineart** with condition strength **0.8**
- **Boundary**: binary mask with Gaussian feathering
- **Backbone**: Stable Diffusion v1.5 + LoRA (rank 8) + dual ControlNet

## Key Features

- Length-weighted outer-frame MAE
- Interior discrete Fréchet distance + Chamfer distance
- CIEDE2000 color fidelity
- TOST equivalence testing
- Runs on NVIDIA RTX 3050 6GB laptop GPU

## Installation

```bash
git clone https://github.com/zx13913865173-Guns/Miao-Embroidery-Phoenix-Mixed-Geometry-Pattern-Generation-via-Stable-Diffusion-and-LoRA-ControlNet.git
cd Miao-Embroidery-Phoenix-Mixed-Geometry-Pattern-Generation-via-Stable-Diffusion-and-LoRA-ControlNet
pip install -r requirements.txt
