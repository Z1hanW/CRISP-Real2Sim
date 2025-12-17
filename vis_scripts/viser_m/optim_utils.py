import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_euler_angles
from pytorch3d.utils import ico_sphere

# Constants
SQRT_EPS = 1e-8
device_default = 'cuda' if torch.cuda.is_available() else 'cpu'


def signed_pow(t: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """Signed power function for superquadric computation."""
    return torch.sign(t) * (torch.abs(t) + SQRT_EPS).pow(exponent)


def parametric_sq_canonical(eta: torch.Tensor, omega: torch.Tensor,
                            eps1: torch.Tensor, eps2: torch.Tensor) -> torch.Tensor:
    """Generate superquadric surface points from parametric angles."""
    eps1 = eps1.clamp(0.1, 2.0)
    eps2 = eps2.clamp(0.1, 2.0)
    cos_eta, sin_eta = torch.cos(eta), torch.sin(eta)
    cos_omega, sin_omega = torch.cos(omega), torch.sin(omega)
    se = signed_pow(sin_eta, eps1)
    ce = signed_pow(cos_eta, eps1)
    so = signed_pow(sin_omega, eps2)
    co = signed_pow(cos_omega, eps2)
    x = ce * so
    y = se
    z = ce * co
    return torch.stack([x, y, z], dim=-1)


def implicit_sq_canonical(points: torch.Tensor,
                          eps1: torch.Tensor, eps2: torch.Tensor) -> torch.Tensor:
    """Compute implicit function value for superquadric."""
    eps1 = eps1.clamp(0.1, 2.0)
    eps2 = eps2.clamp(0.1, 2.0)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    x_term = (torch.abs(x) + SQRT_EPS).pow(2.0 / eps2)
    z_term = (torch.abs(z) + SQRT_EPS).pow(2.0 / eps2)
    y_term = (torch.abs(y) + SQRT_EPS).pow(2.0 / eps1)
    F = (x_term + z_term).pow(eps2 / eps1) + y_term - 1.0
    return F


def create_sq_mesh(eps1: torch.Tensor, eps2: torch.Tensor,
                   scale: torch.Tensor, level: int = 2,
                   device: str = device_default) -> Meshes:
    """Create superquadric mesh using icosphere sampling."""
    ico = ico_sphere(level).to(device)
    verts, faces = ico.get_mesh_verts_faces(0)
    eta = torch.asin(verts[:, 1].clamp(-0.999, 0.999))
    omega = torch.atan2(verts[:, 0], verts[:, 2])
    
    if eps1.dim() == 0:
        eps1 = eps1.view(1).expand(len(eta))
    if eps2.dim() == 0:
        eps2 = eps2.view(1).expand(len(omega))
    
    sq_verts = parametric_sq_canonical(eta, omega, eps1, eps2)
    sq_verts = sq_verts * scale.view(1, 3)
    return Meshes([sq_verts], [faces])


def transform_points_to_body(points_world: torch.Tensor,
                             R_body_to_world: torch.Tensor,
                             T_world: torch.Tensor) -> torch.Tensor:
    """Transform points from world to body coordinates."""
    return (points_world - T_world) @ R_body_to_world.T


def transform_points_to_world(points_body: torch.Tensor,
                              R_body_to_world: torch.Tensor,
                              T_world: torch.Tensor) -> torch.Tensor:
    """Transform points from body to world coordinates."""
    return (R_body_to_world @ points_body.T).T + T_world


def compute_inside_loss(points_world: torch.Tensor,
                        R_body_to_world: torch.Tensor,
                        T_world: torch.Tensor,
                        scale: torch.Tensor,
                        eps1: torch.Tensor,
                        eps2: torch.Tensor) -> torch.Tensor:
    """Compute inside loss for points."""
    pts_body = transform_points_to_body(points_world, R_body_to_world, T_world)
    pts_norm = pts_body / (scale + SQRT_EPS)
    F = implicit_sq_canonical(pts_norm, eps1, eps2)
    return F.relu().mean()


def one_way_chamfer_loss(points: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
    """Compute one-way Chamfer distance."""
    dist = torch.cdist(points, surface)
    return dist.min(dim=1).values.pow(2).mean()


class LMOptimizer:
    """Simplified Levenberg-Marquardt optimizer for superquadric fitting."""
    
    def __init__(self, damping: float = 1e-3):
        self.damping = damping
        self.best_loss = float('inf')
        
    def compute_jacobian_fd(self, loss_fn, params, eps=1e-5):
        """Compute Jacobian using finite differences."""
        loss_0 = loss_fn(params)
        n_params = sum(p.numel() for p in params.values() if p.requires_grad)
        jacobian = []
        
        for name, param in params.items():
            if not param.requires_grad:
                continue
            
            for i in range(param.numel()):
                # Perturb parameter
                original = param.view(-1)[i].item()
                param.view(-1)[i] += eps
                loss_plus = loss_fn(params)
                param.view(-1)[i] = original
                
                # Compute gradient
                grad = (loss_plus - loss_0) / eps
                jacobian.append(grad)
        
        return torch.stack(jacobian) if jacobian else torch.zeros(0)
    
    def step(self, params, gradients, lr_dict):
        """Apply parameter updates with per-parameter learning rates."""
        for name, param in params.items():
            if param.requires_grad and param.grad is not None:
                lr = lr_dict.get(name, 0.01)
                param.data -= lr * param.grad
                
    def update_damping(self, current_loss):
        """Update damping based on loss improvement."""
        if current_loss < self.best_loss:
            self.damping *= 0.5
            self.best_loss = current_loss
        else:
            self.damping *= 2.0


def refine_sq_with_lm(
    results: Dict[str, List[torch.Tensor]],
    lr: float = 2e-3,
    max_iter: int = 300,
    device: str = device_default,
    mesh_level: int = 3,
    optimize_eps: bool = False,
    optimize_scale_z: bool = False,
    optimize_translation_z: bool = False,
    coverage_weight: float = 1e-3,
    inside_weight: float = 0.1,
    eps_regularization: float = 0.5,
    verbose: bool = True,
    loss_threshold: Optional[float] = None,
) -> torch.Tensor:
    """
    Refine superquadrics using Levenberg-Marquardt-inspired optimization.
    
    Args:
        results: Dictionary containing S_items, R_items, T_items, pts_items
        lr: Base learning rate
        max_iter: Maximum iterations
        device: Device for computation
        mesh_level: Level of mesh subdivision
        optimize_eps: Whether to optimize epsilon parameters
        optimize_scale_z: Whether to optimize z-scale
        optimize_translation_z: Whether to optimize z-translation
        coverage_weight: Weight for coverage loss
        inside_weight: Weight for inside loss
        eps_regularization: Weight for epsilon regularization
        verbose: Print progress
        loss_threshold: Drop primitives whose best loss exceeds this value (None disables filtering)
    
    Returns:
        Tensor of refined parameters (N, 11)
    """
    S_items = results["S_items"]
    R_items = results["R_items"]
    T_items = results["T_items"]
    pts_items = results["pts_items"]
    
    # Initialize eps if not present
    if "eps_items" not in results:
        results["eps_items"] = [
            torch.tensor([0.1, 0.1], dtype=torch.float32) for _ in S_items
        ]
    eps_items = results["eps_items"]
    
    N = len(S_items)
    def _pack_params_tensor(
        eps_param_t: torch.Tensor,
        S_log_t: torch.Tensor,
        R_b2w_t: torch.Tensor,
        T_w_t: torch.Tensor,
    ) -> torch.Tensor:
        sx, sy, sz = torch.exp(S_log_t).tolist()
        R_w2b = R_b2w_t.T
        rz, ry, rx = matrix_to_euler_angles(R_w2b.unsqueeze(0), "ZYX").squeeze(0).tolist()
        tx, ty, tz = T_w_t.tolist()
        return torch.tensor([
            eps_param_t[0].item(), eps_param_t[1].item(),
            sx, sy, sz, rz, ry, rx, tx, ty, tz
        ], dtype=torch.float32, device=device)

    params_out: List[torch.Tensor] = []
    keep_indices: List[int] = []
    dropped_indices: List[int] = []
    
    for idx in range(N):
        if verbose:
            print(f"\n=== Refining superquadric {idx+1}/{N} (LM) ===")

        pts_src = pts_items[idx]
        is_tensor = isinstance(pts_src, torch.Tensor)
        is_missing = pts_src is None
        is_empty = is_tensor and pts_src.numel() == 0

        if is_missing or is_empty:
            if verbose:
                reason = "missing" if is_missing else "empty"
                print(f"  Skipping optimization: {reason} point set")

            with torch.no_grad():
                eps_param = eps_items[idx].clone().to(device)
                S_log = S_items[idx].clone().to(device)
                R_b2w = R_items[idx].clone().to(device)
                T_w = T_items[idx].clone().to(device)

                results["eps_items"][idx] = eps_param.cpu()
                results["S_items"][idx] = S_log.cpu()
                results["R_items"][idx] = R_b2w.cpu()
                results["T_items"][idx] = T_w.cpu()

                sx, sy, sz = torch.exp(S_log).tolist()
                R_w2b = R_b2w.T
                rz, ry, rx = matrix_to_euler_angles(R_w2b.unsqueeze(0), "ZYX").squeeze(0).tolist()
                tx, ty, tz = T_w.tolist()

                params_out.append(_pack_params_tensor(eps_param, S_log, R_b2w, T_w))
                keep_indices.append(idx)
            continue

        points_world = pts_src.to(device)

        # Initialize parameters
        eps_param = eps_items[idx].clone().to(device)
        S_log = S_items[idx].clone().to(device).requires_grad_(True)
        R_b2w = R_items[idx].clone().to(device).requires_grad_(True)
        T_w = T_items[idx].clone().to(device).requires_grad_(True)
        
        # Store initial values
        S_log_init = S_log.detach().clone()
        R_b2w_init = R_b2w.detach().clone()
        T_w_init = T_w.detach().clone()
        eps_init = eps_param.detach().clone()
        
        # Set up optimization based on flags
        if optimize_eps:
            eps_param.requires_grad_(True)
        
        # Create masks for selective optimization
        scale_mask = torch.tensor([1., 1., 0. if not optimize_scale_z else 1.], device=device)
        trans_mask = torch.tensor([1., 1., 0. if not optimize_translation_z else 1.], device=device)
        
        # Setup optimizer with different learning rates
        param_groups = [
            {'params': [S_log], 'lr': lr},
            # {'params': [R_b2w], 'lr': lr * 0.5},
            {'params': [T_w], 'lr': lr * 0.5},
        ]
        if optimize_eps:
            param_groups.append({'params': [eps_param], 'lr': lr * 0.1})
        
        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=20, min_lr=1e-5
        )
        
        # Initialize LM components
        lm = LMOptimizer(damping=1e-3)
        best_loss = float('inf')
        best_state = None
        
        for it in range(max_iter):
            optimizer.zero_grad()
            
            # Apply masks to parameters
            S_log_masked = S_log * scale_mask + S_log_init * (1 - scale_mask)
            T_w_masked = T_w * trans_mask + T_w_init * (1 - trans_mask)
            
            # Extract current values
            eps1, eps2 = eps_param[0], eps_param[1]
            scale = torch.exp(S_log_masked)
            
            # Create mesh and transform vertices
            mesh = create_sq_mesh(eps1, eps2, scale, mesh_level, device)
            verts_world = transform_points_to_world(mesh.verts_packed(), R_b2w, T_w_masked)
            
            # Compute losses
            # chamfer = coverage_weight * one_way_chamfer_loss(points_world, verts_world)
            inside = inside_weight * compute_inside_loss(
                points_world, R_b2w, T_w_masked, scale, eps1, eps2
            )
            
            # Regularization terms
            eps_reg = eps_regularization * (eps_param - eps_init).pow(2).sum() if optimize_eps else 0
            scale_reg = 0.1 * (scale - torch.exp(S_log_init)).pow(2).sum()
            rot_reg = 0.05 * (R_b2w - R_b2w_init).pow(2).sum()
            trans_reg = 0.05 * (T_w_masked - T_w_init).pow(2).sum()
            
            # Total loss (using LM-style damping on regularization)
            total_loss = inside  + lm.damping * (eps_reg + scale_reg) # +  # + lm.damping * (eps_reg + scale_reg + rot_reg + trans_reg)
            
            # Backward and optimize
            total_loss.backward()
            
            # Clip gradients
            params_list = [S_log, R_b2w, T_w]
            if optimize_eps:
                params_list.append(eps_param)
            torch.nn.utils.clip_grad_norm_(params_list, 1.0)
            
            optimizer.step()
            
            # Re-orthogonalize rotation matrix
            with torch.no_grad():
                U, _, Vt = torch.linalg.svd(R_b2w, full_matrices=False)
                R_b2w.copy_(U @ Vt)
                
                # Apply masks after optimization step
                S_log.data = S_log.data * scale_mask + S_log_init * (1 - scale_mask)
                T_w.data = T_w.data * trans_mask + T_w_init * (1 - trans_mask)
            
            # Update scheduler and LM damping
            scheduler.step(total_loss)
            lm.update_damping(total_loss.item())
            
            # Track best state
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_state = {
                    'eps_param': eps_param.detach().clone(),
                    'S_log': S_log.detach().clone(),
                    'R_b2w': R_b2w.detach().clone(),
                    'T_w': T_w.detach().clone(),
                    'iteration': it,
                    'inside': inside.item() if isinstance(inside, torch.Tensor) else inside
                }
            
            # Logging
            if verbose and it % 50 == 0:
                print(f"  iter {it:3d} | loss {total_loss:.5f} | "
                      f"inside {inside:.5f} | "
                      f"eps [{eps1:.2f}, {eps2:.2f}] | damping {lm.damping:.2e}")
            
            # Early stopping
            if it > 100 and total_loss.item() < 1e-4:
                if verbose:
                    print(f"  Converged at iter {it}")
                break
        
        # Restore best state
        if best_state is not None:
            eps_param = best_state['eps_param']
            S_log = best_state['S_log']
            R_b2w = best_state['R_b2w']
            T_w = best_state['T_w']
            
            if verbose:
                print(f"  Best@{best_state['iteration']}: "
                      f"inside={best_state['inside']:.5f}")
        
        # Update results
        plane_loss = best_loss if best_state is not None else float('inf')
        drop_plane = False
        loss_value = float(plane_loss)
        if loss_threshold is not None:
            if not math.isfinite(loss_value) or loss_value > loss_threshold:
                drop_plane = True
        if drop_plane:
            dropped_indices.append(idx)
            if verbose:
                readable = loss_value if math.isfinite(loss_value) else float('inf')
                print(f"  Rejecting superquadric {idx+1}: loss={readable:.5f} exceeds threshold {loss_threshold:.5f}")
            continue

        results["eps_items"][idx] = eps_param.cpu()
        results["S_items"][idx] = S_log.cpu()
        results["R_items"][idx] = R_b2w.cpu()
        results["T_items"][idx] = T_w.cpu()
        
        params_out.append(_pack_params_tensor(eps_param, S_log, R_b2w, T_w))
        keep_indices.append(idx)
    
    if N > 0 and len(keep_indices) != N:
        keys_to_filter = ["S_items", "R_items", "T_items", "pts_items", "eps_items"]
        for key in keys_to_filter:
            if key not in results:
                continue
            seq = results[key]
            results[key] = [seq[i] for i in keep_indices if i < len(seq)]
        if verbose:
            print(f"  Filtered out {len(dropped_indices)} primitives above loss threshold.")
    
    if params_out:
        return torch.stack(params_out, dim=0)
    return torch.empty((0, 11), dtype=torch.float32, device=device)


# Alias for compatibility
refine_sq_with_chamfer = refine_sq_with_lm


# Example usage
if __name__ == "__main__":
    # Create sample data
    results = {
        "S_items": [torch.log(torch.tensor([1.0, 1.0, 0.5]))],
        "R_items": [torch.eye(3)],
        "T_items": [torch.zeros(3)],
        "pts_items": [torch.randn(100, 3)]
    }
    
    # Simple usage - just like your original function
    params = refine_sq_with_lm(
        results,
        mesh_level=3,
        device="cuda"
    )
    
    # With optimization flags
    params_advanced = refine_sq_with_lm(
        results,
        mesh_level=3,
        device="cuda",
        optimize_eps=True,         # Enable epsilon optimization
        optimize_scale_z=False,     # Don't optimize z-scale
        optimize_translation_z=False  # Don't optimize z-translation
    )
