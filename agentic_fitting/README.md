# Agentic Superquadric Fitting

This package adds a constrained Codex planning loop around CRISP's deterministic
point-cloud fitter.

The planner sees global point-cloud/SQ renders, residual renders, selected
per-frame 3D observations, RGB frames, and per-frame normal-cluster previews.
It may only return semantic actions. Numeric fitting and acceptance remain
deterministic. Before each planner call, the pipeline also ranks nearby
coplanar and orthogonal primitive pairs by measured support preservation.
Planner actions are evaluated independently in descending-ID order, so a bad
merge is rolled back without discarding good actions from the same plan.
Completion actions collect the local observed neighborhood around a candidate,
allowing a curved cap to support a full sphere even when much of that cap lies
outside the current thin surface primitive.

The reproducible partial-observation regression uses:

```bash
python scripts/create_partial_cap_demo.py --output-dir /data/partial-cap
python -m agentic_fitting.run \
  --pointcloud /data/partial-cap/input/pointcloud_world.npz \
  --baseline-params /data/partial-cap/input/baseline.npz \
  --image-root /data/partial-cap/input/rgb \
  --cluster-root /data/partial-cap/input/clusters \
  --output-dir /data/partial-cap/run
python scripts/verify_partial_cap_demo.py --demo-dir /data/partial-cap
```

After running the three real demos, validate every output contract with:

```bash
python scripts/validate_agentic_outputs.py \
  --demo-root /data/ubuntu/artifacts/crisp-agentic-current \
  --cluster-root /data/ubuntu/artifacts/crisp-agentic/clusters \
  --partial-cap-dir /data/ubuntu/artifacts/crisp-agentic/partial-cap-e2e \
  --output /data/ubuntu/artifacts/crisp-agentic-current/validation.json
```

```bash
python -m agentic_fitting.run \
  --pointcloud /path/to/nksr_input/pointcloud_world.npz \
  --baseline-params /path/to/scene_mesh_sqs/sqs_params.npz \
  --image-root /path/to/sequence/images \
  --cluster-root /path/to/vis/sequence/gv \
  --output-dir /data/ubuntu/artifacts/crisp-agentic/sequence \
  --max-iterations 3 \
  --force
```

Omit `--baseline-params` to run CRISP's existing direct-world-points fitter
first. Outputs include final `sqs_params.npy/.npz`, OBJ pieces, URDF, evidence
images, every Codex event log and plan, and a metric report.
