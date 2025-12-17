# CRISP-Real2Sim

CRISP-Real2Sim contains the complete pipeline we use to turn in-the-wild video into human-scene reconstructions and downstream controllers.
---

## 1. Repository Setup

```bash
git clone --recursive https://github.com/Z1hanW/CRISP-Real2Sim.git
cd CRISP-Real2Sim
```

### Create and Activate the Conda Environment

```bash
conda create -n crisp python=3.10 -y
conda activate crisp
```

### Install PyTorch (CUDA 12.4 build)

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 "xformers>=0.0.27" \
  --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install -r requirements.txt
```

> If you encounter compilation errors (usually on `pytorch3d` or CUDA extensions), install a compatible compiler toolchain: `conda install -c conda-forge gxx_linux-64=11`.

### Extra Installation Scripts

Some dependencies (for rendering, viewers, etc.) are wrapped in helper scripts inside `prep/`:

```bash
cd prep
sh install*
cd ..
```

---

## 2. Download Assets and Data

1. **SMPL/SMPL-X body models** (required for rendering and evaluation)
   - Register at [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/).
   - Place the downloaded `.pkl` files using the structure below.

```
prep/data/
└── body_models/
    ├── smpl/SMPL_{GENDER}.pkl
    └── smplx/SMPLX_{GENDER}.pkl
```

2. **Demo videos and metadata**

```bash
mkdir -p data
gdown --folder "https://drive.google.com/drive/folders/1k712Oj9StmWXRzSeSMiHZc3LtvsVk2Rw" -O data
```

> `gdown` is installed via `requirements.txt`. Use the `-O data` flag so Google Drive folders land under `CRISP-Real2Sim/data`.

---

## 3. Run the Full Pipeline

The scripts expect your source sequences to live under either `*_videos` or `*_img` folders. Remove that suffix when you feed paths to the scripts.

```
data/
├── demo_videos/
│   └── walk-kicking/        # example sequence, this is SEQ_NAME
└── YOUR_videos/
    ├── seq_a/
    └── seq_b/
```

```bash
sh all_gv.sh /path/to/data/demo        # not /path/to/data/demo_videos
```

- The script will iterate through every `*_videos` (or `*_img`) folder under the path you supply.
- Intermediate data, meshes, and evaluations are written back into the respective sequence directories.

---

## 4. Visualize Human–Scene Reconstructions

```bash
sh vis.sh ${SEQ_NAME}
```

Common flags (see script header for the full list):
- `--scene_name`: override the scene used for rendering.
- `--data_root`: custom data directory if not `./data`.
- `--out_dir`: write visualizations to a different folder.

---

## 5. Train Your Agent

Code Testing, See you in days.

---

## 6. Visualize Your Agent

Agent visualization builds on the same `vis.sh` infrastructure:

```bash
python agents/vis_agent.py \
  --checkpoint path/to/checkpoint.pt \
  --seq ${SEQ_NAME} \
  --out_dir outputs/agent_viz/${SEQ_NAME}
```

Pass `--scene_name` or `--camera_pose_file` if your controller requires a custom scene or camera path.

