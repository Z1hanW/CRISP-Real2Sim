Holosoma bridge
---------------
- Converts `results/output/post_scene/*/<hmr_type>/*_ours.npz` into Holosoma-ready SMPLH `.pt` files without touching Holosoma code.
- Outputs land in `holosoma/demo_data/ours_omniretarget/` alongside a merged `holosoma/demo_data/height_dict.pkl`.
- Finger joints that are missing in the SMPL model fall back to the corresponding wrist; object poses are set to identity (good for `robot_only` tasks).

Usage
-----
1. From repo root, run:
   ```
   python holosoma_bridge.py \
     --input-root results/output/post_scene \
     --hmr-type gv \
     --output-root holosoma/demo_data/ours_omniretarget \
     --height-dict holosoma/demo_data/height_dict.pkl
   ```
   Adjust `--hmr-type` if your post_scene data uses a different subfolder name.
2. Run Holosoma retargeting (example for the stairs sequence, robot-only):
   ```
   cd holosoma
   python src/holosoma_retargeting/examples/robot_retarget.py \
     --data_path demo_data/ours_omniretarget \
     --task-type robot_only \
     --task-name 56_outdoor_stairs_up_down \
     --data_format smplh \
     --task-config.object-name ground
   ```
   The generated `height_dict.pkl` is picked up automatically by Holosoma's `calculate_scale_factor`.

Outputs
-------
- `{output-root}/{seq}.pt` : InterMimic-style tensor consumed by `load_intermimic_data`.
- `{output-root}/{seq}_smplh_joints.npz` : debug joints (`global_joint_positions`, `height`, `mocap_framerate`) if you need to inspect or load via another format.
- `height_dict.pkl` : merged subject heights keyed by the prefix of each sequence name (matching Holosoma's expectation).
