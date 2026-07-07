<div align="center">
	<h1>CRISP: Contact-guided Real2Sim from Monocular Video with Planar Scene Primitives</h1>
	<a href="https://arxiv.org/abs/2512.14696"><img src="https://img.shields.io/badge/arXiv-2512.14696-b31b1b" alt="arXiv"></a>
	<a href="https://openreview.net/pdf?id=xlr3NqxUqY"><img src="https://img.shields.io/badge/ICLR_Version-pdf-orange" alt="ICLR Version"></a>
	<a href="https://crisp-real2sim.github.io/CRISP-Real2Sim/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
	<a href="https://drive.google.com/drive/folders/1PX8Pqzqjlh5v0Z6xt-NjzTgpugk4igoN?usp=drive_link"><img src="https://img.shields.io/badge/Video_Dataset-blue" alt="Video Dataset"></a>
</div>
	
![teaser](https://raw.githubusercontent.com/Z1hanW/CRISP-Real2Sim/main/assets/crisp.png)

For any problem you encountered, feel free to raise an issue or email me (zihanwa3@cs.cmu.edu/lucas7eason@gmail.com). 


### [Video Dataset (some Parkours & stairs)](#video-dataset)

**Version note:** this branch/README documents the `v2-by VGGT omega` pipeline.
MegaSAM/TAPIP3D is still kept as a legacy backend, but the v2 path below uses
VGGT-Omega camera/depth priors.

Code pipeline, in one line: scripts `1-8` are `1)` video-to-images convention, `2)` human masks, `3)` improved scene reconstruction, `4)` camera postprocess, `5)` GVHMR, `6)` human-scene alignment and opitmization, `7)` planar fitting, `8)` post-scene alignment + bridge; `MotionTracking` then handles RL train/eval/viser.

---

### 1. Repository Setup

```bash
git clone --recursive https://github.com/Z1hanW/CRISP-Real2Sim.git
cd CRISP-Real2Sim
bash scripts/manage_submodules.sh init
bash setups/setup_crisp.sh
conda activate crisp
```

External code dependencies under `prep/` and `real2sim2real/` are managed as
submodules. Use `bash scripts/manage_submodules.sh status` to inspect recorded
commits, and `bash scripts/manage_submodules.sh update` only when you
intentionally want to move submodule pointers forward.


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

For the `v2-by VGGT omega` path, initialize the `prep/vggt-omega` submodule,
set `SCENE_RECON_BACKEND=vggt_omega`, and point CRISP at the local checkpoint.
The VGGT-Omega checkpoints are gated Hugging Face files: request access at
https://huggingface.co/facebook/VGGT-Omega first, wait for approval, then
authenticate locally and download the checkpoint outside the repo. Do not commit
Hugging Face tokens or write them into scripts.

```bash
bash scripts/manage_submodules.sh init
huggingface-cli login
mkdir -p /path/to/models/vggt-omega
huggingface-cli download facebook/VGGT-Omega vggt_omega_1b_512.pt \
  --local-dir /path/to/models/vggt-omega

export SCENE_RECON_BACKEND=vggt_omega
export VGGT_OMEGA_REPO="$PWD/prep/vggt-omega"
export VGGT_OMEGA_CHECKPOINT=/path/to/models/vggt-omega/vggt_omega_1b_512.pt
bash run_crisp_video.sh /path/to/data/demo
```

If this setup is not correct, the VGGT step stops early with an error like:

```text
[vggt_omega] ERROR: VGGT_OMEGA_CHECKPOINT is required.
[vggt_omega] ERROR: request access at https://huggingface.co/facebook/VGGT-Omega, then download the checkpoint and export VGGT_OMEGA_CHECKPOINT=/path/to/model.pt
```

For Hugging Face 401/403 errors, confirm that the account used by
`huggingface-cli login` has already been approved for the gated
`facebook/VGGT-Omega` repo.

The VGGT-Omega adapter writes the same CRISP raw camera/depth prior file used by
the rest of the pipeline:

```text
results/init/vslam/raw_vggt_omega_priors/<SEQ_NAME>.npz
```


The `v2-by VGGT omega` path writes backend-tagged scene outputs:

```text
results/output/scene/
└── <SEQ_NAME>_vggt_omega_gv_sgd_cvd_hr.npz

results/output/scene_vggt_omega_consistent_camera_min1/
└── <SEQ_NAME>/gv/scene_mesh_sqs/
    ├── scene_mesh_sqs.urdf
    └── ...

results/output/post_scene_vggt_omega/        # if the postprocess wrapper is run
└── <SEQ_NAME>/gv/
    ├── hmr/human_motion.npz
    ├── scene_mesh_sqs/
    └── ...
```

Comment: `results/output/scene` stores the aligned scene `.npz`;
`scene_vggt_omega_consistent_camera_min1` stores the VGGT-Omega SQS artifacts;
`post_scene_vggt_omega` is the rotated z-up post-processed version used for
bridging into MotionTracking.

#### Planar Primitive Fitting Notes

Step 7 fits watertight scene primitives from the fused 2.5D point cloud. The
primitive geometry should faithfully reflect the saved point cloud; do not hide
bad fits with viewer-only offsets, scales, or clamps.

Current support-aware fitting rules:

- Fit each plane in scene coordinates. Local min-area rectangles are only the
  initial estimate; for repeated same-normal surfaces, such as stair platforms,
  the final fit snaps to a normal-family consensus in-plane axis when that axis
  has enough votes and the support/area tradeoff is bounded. This avoids one
  partially observed platform choosing a slanted axis that protrudes into a
  neighboring surface.
- After the axis is selected, trim weakly supported outer footprint edges with an
  occupancy grid before exporting SQS pieces.
- Merge only same-layer, normal-similar segment candidates before fitting.
- Split a fitted plane when support is poor, when there is a clear footprint gap,
  or when a large rectangle wastes area relative to two simpler guillotine pieces.
  The large-envelope split is intentionally conservative: by default it only
  tests planes above `8.0 m^2`, uses a fine support threshold of `0.96`, and
  accepts the split only when total fitted area drops by at least `4.5%` or the
  support objective clearly improves.
- For the experimental mesh-piece postprocess in
  `vis_scripts/viser_m/convex_clip_sqs_primitives.py`, only spatially adjacent
  neighboring non-parallel surfaces may act as infinite-plane cutters. The
  finite line intervals on both footprints must overlap or be close
  (`--neighbor-max-line-gap`), and the two pieces' observed support points must
  be near each other (`--neighbor-max-support-distance`). Use
  `--neighbor-snap-fill` for the alternative "make adjacent surfaces meet"
  experiment: it disables support-cover shrinking and snaps expanded footprints
  to spatially adjacent shared-edge lines while preserving almost all support
  points. Snap-fill is selective: a candidate replaces the original piece only
  when its support does not drop and its area stays within
  `--snap-fill-max-area-ratio`; rejected pieces copy the original mesh instead
  of being re-exported. If a true surface footprint is concave, enable
  `--support-piece-cover` so the supported cells are covered by a small set of
  convex mesh pieces instead of one convex hull that protrudes. In these
  mesh-piece modes `scene_mesh_sqs.obj` and `pieces/*.obj` are the faithful
  geometry; `sqs_params.npy/.npz` are compatibility metadata only.
- Keep generated SQS, point clouds, `world_rotation`, and `shared_translation`
  in the same frame. For debugging, transform data into z-up using the saved
  post-scene transform instead of changing the viewer camera/up axis.

Stairs snap-fill/selective viewer provenance:

- Current accepted all115 rerun, logged on 2026-06-28:
  - Purpose: test whether the snap-fill/selective algorithm that worked on
    `stair_75` also transfers to the other stair sequences.
  - Use the visualizer/fit-points upstream path, not the direct-world-points
    fitting path.
  - Combined upstream root:
    `/tmp/crisp_stairs_same75_visualizer_upstream_all115`
  - Upstream components:
    `/tmp/crisp_stairs_legacy_stair75_112` and
    `/tmp/crisp_stairs_legacy_stair75_outdoor3`
  - Postprocess output root:
    `/tmp/crisp_stairs_same75_post_visualizer_all115`
  - z-up compare root:
    `/tmp/crisp_stairs_same75_post_visualizer_all115_zup`
  - Active all115 viewer:
    `http://localhost:9329`
  - Strict 10-sequence reference viewer:
    `http://localhost:9328`
- The 2026-06-28 all115 rerun completed 115/115 sequences. The postprocess
  produced 3999 pieces total, with 190 total cuts and 81 snap-fill-accepted
  pieces. For `stair_75`, the all115 rerun has 36 v2 pieces, 3 cuts, and 2
  snap-fill-accepted pieces. Compared with the previously accepted
  `stair_75` postprocess output, the rerun differs only at floating-point
  export precision (`sqs_params` max abs diff about `2.15e-6`, mesh vertex max
  abs diff about `1.91e-6`, faces identical).
- The snap-fill/selective postprocess command family used for this accepted
  run is:

```bash
/home/ubuntu/miniconda3/envs/crisp/bin/python \
  vis_scripts/viser_m/convex_clip_sqs_primitives.py \
  --input-seq-root <UPSTREAM_ROOT>/<SEQ>/gv \
  --output-seq-root <POST_ROOT>/<SEQ>/gv \
  --max-points 1400000 \
  --grid-base 80 \
  --max-cuts 4 \
  --min-points 500 \
  --min-area-reduction 0.025 \
  --min-support-gain 0.02 \
  --min-keep-fraction 0.94 \
  --target-support 0.985 \
  --z-margin 0.025 \
  --cut-padding 0.015 \
  --neighbor-plane-clip \
  --neighbor-min-angle-deg 15.0 \
  --neighbor-max-cuts 4 \
  --neighbor-min-keep-fraction 0.78 \
  --neighbor-min-area-reduction 0.015 \
  --neighbor-support-drop 0.025 \
  --neighbor-footprint-margin 0.08 \
  --neighbor-priority \
  --neighbor-spatial-filter \
  --neighbor-max-line-gap 0.25 \
  --neighbor-max-support-distance 0.25 \
  --neighbor-support-sample-points 4096 \
  --neighbor-snap-fill \
  --snap-fill-expand-margin 0.05 \
  --snap-fill-max-lines 4 \
  --snap-fill-max-discard-fraction 0.03 \
  --snap-fill-min-area-ratio 0.55 \
  --snap-fill-min-final-area-ratio 0.8 \
  --snap-fill-max-area-ratio 1.02 \
  --snap-fill-max-support-drop 0.005 \
  --force
```

- In this version, the faithful geometry is `scene_mesh_sqs.obj` plus
  `scene_mesh_sqs/pieces/*.obj`. The `sqs_params.npy/.npz` files are kept for
  compatibility and bookkeeping; do not judge the final mesh-piece shape from
  primitive parameters alone.
- The output should carry the mesh-postprocess metadata keys
  `convex_clip_mesh_only`, `surface_piece_cover`, and `convex_clip_note`.
  These indicate the adjacent-surface snap-fill/selective support-cover pass was
  applied.
- The accepted all115 viewer command is:

```bash
/home/ubuntu/miniconda3/envs/crisp/bin/python -u \
  vis_scripts/viser_m/visualizer_sqs_v2_compare.py \
  --baseline-root /tmp/crisp_stairs_same75_post_visualizer_all115_zup/baseline \
  --v2-root /tmp/crisp_stairs_same75_post_visualizer_all115_zup/v2 \
  --hmr-type gv \
  --initial-seq stair_75 \
  --port 9329 \
  --max-points 200000 \
  --side-offset 6.0 \
  --sequences <ALL_115_SEQUENCES>
```

- Superseded note: do not use the old mixed-root viewer on port 9325 to judge
  whether this algorithm generalizes. That older root,
  `/tmp/crisp_stairs_sqs_snapfill_selective_fitpoints_all115_fixed_zup`, mixed
  10 old-style fit-points sequences with 105 direct-world-points fit outputs.
  It was useful for inspection but was not a homogeneous all115 rerun of the
  `stair_75`-approved path.

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

### 6. HMR-to-G1 / GMR Scale Notes

When debugging G1 retargeting from CRISP HMR outputs, keep the coordinate and
scale rules explicit. The final aligned HMR track under `post_scene` is already
in the display/scene frame. Feed that HMR result directly to GMR; do not add an
extra HMR-to-G1 root alignment pass afterward.

Current expected behavior:

- `prepare_scene_frame_gmr_inputs.py` should pass the final HMR joints in scene
  coordinates to GMR.
- `run_retargeting_backend.py` should save direct GMR qpos with
  `root_alignment_mode=none_direct_gmr_from_hmr_joints`.
- Direct GMR qpos should not contain `root_alignment_uniform_scale`.
- The combined HMR/G1 viewer should use `robot_position_scale=1.0` for direct
  qpos, `mesh_scale=1.0`, no `z_offset`, and no fixed `world_t`.

Do not use viewer-side scale to make G1 match the human height. G1 is physically
smaller than the human, so its mesh should stay at the robot's native size. If a
legacy qpos contains `root_alignment_uniform_scale`, it came from an older extra
alignment path; treat it as legacy/debug data rather than the default output.
The viewer must faithfully reflect the saved data. Do not use viewer-only
offsets, scale factors, ground clamps, or other visual fixes to hide retargeting
or reconstruction problems; fix the upstream data generation path instead.

Important GMR caveat: upstream GMR's `scale_human_data()` scales the root
absolute position as well as local body offsets according to the IK
`human_scale_table`. That can make G1 appear lower or shifted relative to the
original HMR in scene coordinates. This is GMR-internal behavior, not a viewer
offset. If scene-fixed root placement is needed, change GMR's scaling rule so
only body offsets relative to the pelvis are scaled; do not compensate by
viewer-side scale or translation.

### 7. Holosoma Submodule And Environment

Holosoma retargeting is pulled through the `real2sim2real` submodule, which
tracks `https://github.com/Z1hanW/holosoma-crisp.git` on `main`. Keep CRISP
overrides committed in that fork/branch, then commit the updated submodule
pointer in this repo. Do not rely on uncommitted edits in an external
`/home/ubuntu/FAR/holosoma` checkout.

To sync future upstream Holosoma changes, update inside the submodule first,
then update the parent pointer:

```bash
git submodule update --init --recursive real2sim2real
git -C real2sim2real checkout main
git -C real2sim2real remote add upstream https://github.com/Z1hanW/holosoma.git  # first time only
git -C real2sim2real fetch upstream
git -C real2sim2real rebase upstream/main
git -C real2sim2real push --force-with-lease origin main
git add real2sim2real .gitmodules
git commit -m "Update holosoma submodule"
```

`run_retargeting_backend.py` resolves Holosoma in this order:
`HOLOSOMA_RETARGETING_ROOT`, `HOLOSOMA_ROOT`, the `real2sim2real` submodule,
then `/home/ubuntu/FAR/holosoma` as a local fallback. It also prepends the
resolved submodule package root to `PYTHONPATH` when invoking Holosoma. The
default Holosoma environment is `HOLOSOMA_ENV=hsretargeting`, using
`HOLOSOMA_CONDA_EXE` or `$HOME/.holosoma_deps/miniconda3/bin/conda` when
available.

---

### 8. Train Your Agent
```bash
cd MotionTracking
```

See [MotionTracking/README.md](MotionTracking/README.md).

That guide covers environment setup, CRISP-to-RL transfer, training, `viser`
debug runs, evaluation, and SMPL parameter export. The commands there assume
your working directory is already `MotionTracking`.

---

### 8. Visualize Your Agent

Agent visualization builds on the same `vis.sh` infrastructure:

```bash
python agents/vis_agent.py \
  --checkpoint path/to/checkpoint.pt \
  --seq ${SEQ_NAME} \
  --out_dir outputs/agent_viz/${SEQ_NAME}
```

Pass `--scene_name` or `--camera_pose_file` if your controller requires a custom scene or camera path.

---

### 9. Optional NKSR Surface Reconstruction

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
@inproceedings{wangcontact,
title={Contact-guided Real2Sim from Monocular Video with Planar Scene Primitives},
author={Wang, Zihan and Wang, Jiashun and Tan, Jeff and Zhao, Yiwen and Hodgins, Jessica K and Tulsiani, Shubham and Ramanan, Deva},
booktitle={The Fourteenth International Conference on Learning Representations}
}
```

## Acknowledgment

We thank [viser](https://github.com/viser-project/viser) for supporting our visualization workflow.
