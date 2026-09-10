<div align="center">
	<h1>CRISP: Contact-guided Real2Sim from Monocular Video with Planar Scene Primitives</h1>
	<a href="https://arxiv.org/abs/2512.14696"><img src="https://img.shields.io/badge/arXiv-2512.14696-b31b1b" alt="arXiv"></a>
	<a href="https://openreview.net/pdf?id=xlr3NqxUqY"><img src="https://img.shields.io/badge/ICLR_Version-pdf-orange" alt="ICLR Version"></a>
	<a href="https://crisp-real2sim.github.io/CRISP-Real2Sim/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
	<a href="https://drive.google.com/drive/folders/1PX8Pqzqjlh5v0Z6xt-NjzTgpugk4igoN?usp=drive_link"><img src="https://img.shields.io/badge/Video_Dataset-blue" alt="Video Dataset"></a>
</div>

**Zihan Wang, Jiashun Wang, Jeff Tan, Yiwen Zhao, Jessica Hodgins, Shubham Tulsiani, Deva Ramanan. International Conference on Learning Representations (2026).**

## Research overview

CRISP reconstructs human motion and static, rigid scene geometry from monocular RGB video, using planar primitives and human-scene contacts to produce simulation-ready assets.

In [Table 1](https://arxiv.org/html/2512.14696v3#S4.T1), CRISP and VideoMimic assets use the same RL tracking pipeline: overall success is 93.1% versus 44.8%, and RL throughput is 23K versus 16K FPS. FPS includes simulation and policy learning. The paper does not evaluate moving-object or deformable-object reconstruction; curved surfaces and inaccurate motion/contact estimates remain limitations.

	
![teaser](https://raw.githubusercontent.com/Z1hanW/CRISP-Real2Sim/main/assets/crisp.png)

For any problem you encountered, feel free to raise an issue or email me (zihanwa3@cs.cmu.edu/lucas7eason@gmail.com). 


### [Video Dataset (some Parkours & stairs)](#video-dataset)

Code pipeline, in one line: scripts `1-8` are `1)` video-to-images convention, `2)` human masks, `3)` improved scene reconstruction, `4)` camera postprocess, `5)` GVHMR, `6)` human-scene alignment and opitmization, `7)` planar fitting, `8)` post-scene alignment + bridge; `MotionTracking` then handles RL train/eval/viser.

---

### 1. Repository Setup

```bash
git clone --recursive https://github.com/Z1hanW/CRISP-Real2Sim.git
cd CRISP-Real2Sim
bash setups/setup_crisp.sh
conda activate crisp
```


Optional demo shortcut: [`run_demo.sh`](setups/run_demo.sh), one trick I found is to launch codex --yolo / claude code inside of this repo and ask it to set up environment, it can help with lots of conflicts among different machines. 

---

### 2. Download Assets and Data

See [prep/README.md](prep/README.md) for the full preparation flow:

- SMPL / SMPL-X body models
- demo videos and metadata
- optional contact hallucination assets

---

### 3. Run the Full Pipeline

The wrapper and scripts expect your source sequences to live under either
`*_videos` or `*_img` folders. Remove that suffix when you feed paths to the
scripts.

```text
data/
├── demo_videos/
│   └── wall-kicking.mp4
└── YOUR_videos/
    ├── seq_a.mp4
    └── seq_b.mp4
```

For your own data:

```bash
bash run_crisp_video.sh /path/to/data/demo        # not /path/to/data/demo_videos
```


Results will contain both `scene` and `post_scene`:

```text
results/output/scene/
├── <SEQ_NAME>_gv_sgd_cvd_hr.npz
└── <SEQ_NAME>/gv/scene_mesh_sqs/
    ├── scene_mesh_sqs.urdf
    └── ...

results/output/post_scene/
└── <SEQ_NAME>/gv/
    ├── hmr/human_motion.npz
    ├── scene_mesh_sqs/
    └── ...
```

Comment: `scene` is the direct CRISP reconstruction output; `post_scene` is the
aligned, rotated z-up post-processed version used for bridging into MotionTracking.

---

### 4. Contact Hallucination (Optional)

See [prep/README.md](prep/README.md#2-optional-contact-hallucination) for the
full contact setup and data-prep details.

```bash
bash scripts/0_interactvlm.sh /abs/path/to/data/demo/pkr stairs
```

If you want a single batch entry with contact hallucination included:

```bash
bash scripts/all_gv_contact.sh /abs/path/to/data/demo stairs
```

---

### 5. Visualize Human–Scene Reconstructions

Compile viser if needed:

```bash
cd vis_scripts/viser_m
pip install -e .
```

Visualize your sequences:

```bash
bash vis.sh ${SEQ_NAME}
```

If you also ran the optional Contact Hallucination step:

```bash
USE_CONTACT=on bash vis.sh ${SEQ_NAME}
```

Common flags (see script header for the full list):
- `--scene_name`: override the scene used for rendering.
- `--data_root`: custom data directory if not `./data`.
- `--out_dir`: write visualizations to a different folder.

---

### 6. Train Your Agent
```bash
cd MotionTracking
```

See [MotionTracking/README.md](MotionTracking/README.md).

That guide covers environment setup, CRISP-to-RL transfer, training, `viser`
debug runs, evaluation, and SMPL parameter export. The commands there assume
your working directory is already `MotionTracking`.

---

### 7. Visualize Your Agent

Agent visualization builds on the same `vis.sh` infrastructure:

```bash
python agents/vis_agent.py \
  --checkpoint path/to/checkpoint.pt \
  --seq ${SEQ_NAME} \
  --out_dir outputs/agent_viz/${SEQ_NAME}
```

Pass `--scene_name` or `--camera_pose_file` if your controller requires a custom scene or camera path.

---

### 8. Optional NKSR Surface Reconstruction

If you want a more detailed surface and want to test NKSR on CRISP point
clouds, install NKSR in a cloned `crisp` environment:

```bash
bash setups/setup_crisp_nksr.sh
conda activate crisp_nksr
```

Then convert the saved CRISP point cloud to an NKSR mesh:

```bash
cd vis_scripts/viser_m
NKSR_MAX_INPUT_POINTS=200000 NKSR_DETAIL_LEVEL=0.1 bash run_nksr.sh ${SEQ_NAME}
```

and writes in:

```text
results/output/scene/<SEQ_NAME>/gv/nksr
```

Comment: this is an extra detailed-surface test path; the main CRISP pipeline
does not depend on NKSR.

---

## Video Dataset

We release a curated and clipped video dataset here:
[Video Dataset](https://drive.google.com/drive/folders/1PX8Pqzqjlh5v0Z6xt-NjzTgpugk4igoN?usp=drive_link).

It includes both self-captured videos and internet videos we collect with
hours efforts. A substantial portion of these videos currently fail in CRISP because HMR is still not
reliable under high-dynamics motion. We still decided to release them because
we know that finding clean suitable videos is a real bottleneck for
such a real2sim pipeline.

It also includes videos related to [PROX](https://prox.is.tue.mpg.de/),
[EMDB](https://eth-ait.github.io/emdb/), and
[RICH](https://rich.is.tue.mpg.de/), please consider citing them and CRISP if you find those video data are useful for your work.

---

## Citation

If the idea, code, visualization, or video data are helpful for your research,
please consider citing CRISP.

```bibtex
@inproceedings{wang2026contact,
  title={Contact-guided real2sim from monocular video with planar scene primitives},
  author={Wang, Zihan and Wang, Jiashun and Tan, Jeff and Zhao, Yiwen and Hodgins, Jessica and Tulsiani, Shubham and Ramanan, Deva},
  booktitle={International Conference on Learning Representations},
  volume={2026},
  pages={115452--115468},
  year={2026}
}
```

## Acknowledgment

We thank [viser](https://github.com/viser-project/viser) for supporting our visualization workflow.
