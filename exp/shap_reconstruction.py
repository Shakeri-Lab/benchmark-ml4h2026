from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import time

# -----------------------------------------------------------------------------
# Unchanged / Helper Functions (Kept for compatibility or reduced usage)
# -----------------------------------------------------------------------------

def _masked_regions(mask_t: np.ndarray) -> List[Tuple[int, int]]:
    """Legacy helper, largely superseded by vectorized logic in reconstruct_inplace."""
    regions: List[Tuple[int, int]] = []
    in_mask = False
    start = 0
    for t, m in enumerate(mask_t):
        if m and not in_mask:
            in_mask = True
            start = t
        elif (not m) and in_mask:
            regions.append((start, t))
            in_mask = False
    if in_mask:
        regions.append((start, len(mask_t)))
    return regions

def _combine_mask(mask_bt: np.ndarray, ch: Sequence[int], mode: str = "all") -> np.ndarray:
    if not ch:
        return np.zeros(mask_bt.shape[1], dtype=bool)
    sub = mask_bt[:, ch]
    if sub.ndim == 1:
        return sub
    if mode == "any":
        return np.any(sub, axis=1)
    return np.all(sub, axis=1)

def _infer_meal_type(
    meal_vals: np.ndarray,
    meal_binary_max: float,
    meal_binary_ratio: float,
    override: Optional[str],
) -> str:
    if override in {"binary", "grams"}:
        return override
    if meal_vals.size == 0:
        return "grams"
    if meal_vals.max() <= meal_binary_max and np.mean(meal_vals <= meal_binary_max) >= meal_binary_ratio:
        return "binary"
    return "grams"

def compute_reconstruction_stats(
    windows: Sequence[np.ndarray],
    modality_indices: Mapping[str, Sequence[int]],
    recon_cfg: Any,
) -> Dict[str, Any]:
    meal_idxs = list(modality_indices.get("meal", []))
    bolus_idxs = list(modality_indices.get("bolus", []))

    meal_vals: List[float] = []
    bolus_vals: List[float] = []
    for w in windows:
        if w.ndim != 2:
            continue
        if meal_idxs:
            mv = w[:, meal_idxs].reshape(-1)
            mv = mv[mv > 0]
            if mv.size:
                meal_vals.append(mv)
        if bolus_idxs:
            bv = w[:, bolus_idxs].reshape(-1)
            bv = bv[bv > 0]
            if bv.size:
                bolus_vals.append(bv)

    meal_all = np.concatenate(meal_vals) if meal_vals else np.asarray([], dtype=np.float32)
    bolus_all = np.concatenate(bolus_vals) if bolus_vals else np.asarray([], dtype=np.float32)

    meal_type = _infer_meal_type(
        meal_all,
        getattr(recon_cfg, "meal_binary_max", 1.5),
        getattr(recon_cfg, "meal_binary_ratio", 0.95),
        getattr(recon_cfg, "meal_type", None),
    )
    median_meal = float(np.median(meal_all)) if meal_all.size else 0.0
    median_bolus = float(np.median(bolus_all)) if bolus_all.size else 0.0

    return {
        "meal_type": meal_type,
        "median_meal": median_meal,
        "median_bolus": median_bolus,
    }

# -----------------------------------------------------------------------------
# Optimized Vectorized Reconstruction
# -----------------------------------------------------------------------------

def _fill_cgm_vectorized(x: torch.Tensor, mask: torch.Tensor) -> None:
    """
    Vectorized Linear Interpolation.
    x: (N, T) values
    mask: (N, T) bool, True where missing
    """
    if x.numel() == 0:
        return

    B, T = x.shape
    device = x.device
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

    # 1. Identify previous valid index (Forward pass with cummax)
    # Mark invalid positions with -1, valid with their index
    prev_idx_candidates = idx.clone()
    prev_idx_candidates[mask] = -1
    # Cummax propagates the largest index seen so far (the previous valid one)
    prev_idx = prev_idx_candidates.cummax(dim=1).values

    # 2. Identify next valid index (Backward pass with cummin)
    # Mark invalid positions with T, valid with their index
    next_idx_candidates = idx.clone()
    next_idx_candidates[mask] = T
    # Cummin on flipped array propagates the smallest index seen (the next valid one)
    next_idx = next_idx_candidates.flip(dims=[1]).cummin(dim=1).values.flip(dims=[1])

    # 3. Gather values
    # Clamp to handle edge cases (-1 or T) safely during gather
    gather_prev = prev_idx.clamp(0, T - 1)
    gather_next = next_idx.clamp(0, T - 1)
    
    val_prev = torch.gather(x, 1, gather_prev)
    val_next = torch.gather(x, 1, gather_next)

    # 4. Compute Interpolation
    # Cases:
    # A. Left & Right OK: Interpolate
    # B. Only Left OK: Constant fill forward
    # C. Only Right OK: Constant fill backward
    # D. Neither OK: Fill 0.0
    
    left_ok = prev_idx != -1
    right_ok = next_idx != T
    both_ok = left_ok & right_ok
    
    # Linear weights
    dist = (next_idx - prev_idx).float()
    # Avoid div by zero (masked only happens where dist >= 2, unmasked dist=0 but unused)
    # We use a safe denominator
    dist = dist.masked_fill(dist == 0, 1.0)
    alpha = (idx - prev_idx).float() / dist

    interp_vals = val_prev + (val_next - val_prev) * alpha

    # Apply logic to mask only
    # Note: We must ensure we don't overwrite valid data (though mask handles that)
    
    # Create final fill tensor
    # Default to 0.0
    fill_values = torch.zeros_like(x)
    
    # Apply conditions
    # Case A: Both OK
    fill_values = torch.where(both_ok, interp_vals, fill_values)
    # Case B: Left Only (Forward Fill)
    fill_values = torch.where(left_ok & (~right_ok), val_prev, fill_values)
    # Case C: Right Only (Backward Fill)
    fill_values = torch.where((~left_ok) & right_ok, val_next, fill_values)
    
    # Assign back to x where masked
    x.masked_scatter_(mask, fill_values[mask])


def _fill_basal_vectorized(x: torch.Tensor, mask: torch.Tensor) -> None:
    """
    Vectorized Midpoint Fill.
    x: (N, T)
    mask: (N, T)
    """
    if x.numel() == 0:
        return

    B, T = x.shape
    device = x.device
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

    # 1. Previous and Next indices (same as CGM)
    prev_idx_candidates = idx.clone()
    prev_idx_candidates[mask] = -1
    prev_idx = prev_idx_candidates.cummax(dim=1).values

    next_idx_candidates = idx.clone()
    next_idx_candidates[mask] = T
    next_idx = next_idx_candidates.flip(dims=[1]).cummin(dim=1).values.flip(dims=[1])

    # 2. Gather Values
    gather_prev = prev_idx.clamp(0, T - 1)
    gather_next = next_idx.clamp(0, T - 1)
    val_prev = torch.gather(x, 1, gather_prev)
    val_next = torch.gather(x, 1, gather_next)

    # 3. Compute Midpoint Logic
    # Midpoint index calculation: (s + e) // 2
    # In our index space: s = prev_idx + 1, e = next_idx
    # Formula: (prev_idx + next_idx + 1) // 2
    mid_idx = (prev_idx + next_idx + 1) // 2
    
    # If current t < mid, use left. Else use right.
    use_left_val = idx < mid_idx
    
    # 4. Apply Conditions
    left_ok = prev_idx != -1
    right_ok = next_idx != T
    
    fill_values = torch.zeros_like(x)
    
    # Case A: Both OK -> Step function
    step_val = torch.where(use_left_val, val_prev, val_next)
    fill_values = torch.where(left_ok & right_ok, step_val, fill_values)
    
    # Case B: Left Only
    fill_values = torch.where(left_ok & (~right_ok), val_prev, fill_values)
    
    # Case C: Right Only
    fill_values = torch.where((~left_ok) & right_ok, val_next, fill_values)

    x.masked_scatter_(mask, fill_values[mask])


def reconstruct_inplace(
    x_masked: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    modality_channel_indices: Mapping[str, Sequence[int]],
    recon_cfg: Any,
    stats: Mapping[str, Any],
    touched_modalities: Optional[Sequence[str]] = None,
    profiler: Optional[Dict[str, float]] = None,
) -> None:
    if x_masked.numel() == 0:
        return

    # Ensure tensors on correct device
    device = x_masked.device
    if not isinstance(mask, torch.Tensor):
        mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool)
    else:
        mask_t = mask.to(device=device, dtype=torch.bool)

    if mask_t.ndim == 2:
        mask_t = mask_t.unsqueeze(0).expand(x_masked.shape[0], -1, -1)
    
    if mask_t.shape != x_masked.shape:
        raise ValueError(f"mask shape {tuple(mask_t.shape)} does not match x shape {tuple(x_masked.shape)}")

    # Parse touched modalities
    touched = None if touched_modalities is None else {str(m) for m in touched_modalities}
    def _do(mod: str) -> bool:
        return touched is None or mod in touched

    # Indices
    cgm_idxs = list(modality_channel_indices.get("cgm", [])) if _do("cgm") else []
    basal_idxs = list(modality_channel_indices.get("basal", [])) if _do("basal") else []
    
    do_meal = _do("meal")
    do_bolus = _do("bolus")
    do_meal_bolus = do_meal or do_bolus
    meal_idxs = list(modality_channel_indices.get("meal", [])) if do_meal_bolus else []
    bolus_idxs = list(modality_channel_indices.get("bolus", [])) if do_meal_bolus else []

    def _t_start(key: str) -> float | None:
        if profiler is None:
            return None
        if x_masked.is_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _t_end(key: str, t0: float | None) -> None:
        if t0 is None:
            return
        if x_masked.is_cuda:
            torch.cuda.synchronize()
        profiler[key] = profiler.get(key, 0.0) + (time.perf_counter() - t0)

    # --- 1. CGM Reconstruction (Batch * Channel parallel) ---
    if cgm_idxs:
        t0 = _t_start("recon_cgm")
        # Reshape to (Batch * Channels, Time) for efficient 2D processing
        x_cgm = x_masked[:, :, cgm_idxs].transpose(1, 2).reshape(-1, x_masked.shape[1])
        m_cgm = mask_t[:, :, cgm_idxs].transpose(1, 2).reshape(-1, x_masked.shape[1])
        _fill_cgm_vectorized(x_cgm, m_cgm)
        # Scatter back (view shares memory, so just reshape logic is conceptual, 
        # but masked_scatter_ acts on original memory if done right. 
        # However, advanced indexing usually creates copies. 
        # We need to assign back carefully.)
        
        # Optimization: _fill_cgm_vectorized modifies in-place.
        # But `x_masked[:,:,idxs]` creates a copy if idxs is a list.
        # We need to process and copy back.
        
        # Let's do it cleanly:
        # Extract slices, Modify, Assign back.
        # Ideally, we pass the slice if it's contiguous, but fancy indexing isn't.
        # So we process the copy and assign back.
        
        # To avoid massive memory overhead, we can process all batch/channels in one simplified tensor
        # if we construct it carefully.
        
        # Re-extract to ensure we can write back
        x_cgm_orig = x_masked[:, :, cgm_idxs]
        m_cgm_orig = mask_t[:, :, cgm_idxs]
        
        # Flatten batch and channel dims
        B_dim, T_dim, C_dim = x_cgm_orig.shape
        x_flat = x_cgm_orig.transpose(1, 2).reshape(-1, T_dim)
        m_flat = m_cgm_orig.transpose(1, 2).reshape(-1, T_dim)
        
        _fill_cgm_vectorized(x_flat, m_flat)
        
        # Write back
        x_masked[:, :, cgm_idxs] = x_flat.view(B_dim, C_dim, T_dim).transpose(1, 2)
        _t_end("recon_cgm", t0)

    # --- 2. Basal Reconstruction ---
    if basal_idxs:
        t0 = _t_start("recon_basal")
        x_basal_orig = x_masked[:, :, basal_idxs]
        m_basal_orig = mask_t[:, :, basal_idxs]
        
        B_dim, T_dim, C_dim = x_basal_orig.shape
        x_flat = x_basal_orig.transpose(1, 2).reshape(-1, T_dim)
        m_flat = m_basal_orig.transpose(1, 2).reshape(-1, T_dim)
        
        _fill_basal_vectorized(x_flat, m_flat)
        
        x_masked[:, :, basal_idxs] = x_flat.view(B_dim, C_dim, T_dim).transpose(1, 2)
        _t_end("recon_basal", t0)

    # --- 3. Meal / Bolus Reconstruction ---
    if not (meal_idxs or bolus_idxs):
        return

    k = int(getattr(recon_cfg, "meal_bolus_k", 3))
    meal_type = stats.get("meal_type", "grams")
    median_meal = float(stats.get("median_meal", 0.0))
    median_bolus = float(stats.get("median_bolus", 0.0))
    meal_impulse = 1.0 if meal_type == "binary" else median_meal
    bolus_impulse = median_bolus

    # Prepare logic for cross-modality impulse injection
    # We operate on (Batch, Time) logic for the presence of events
    
    # Helper to clean masks and events
    def get_region_mask_and_clear(idxs, target_x, target_m):
        if not idxs:
            return None
        if not target_m[:, :, idxs].any():
            return None
        # Zero out masked regions
        # target_x[:, :, idxs] -> (B, T, C). Mask -> (B, T, C)
        # We need to zero where mask is True
        for i in idxs:
            target_x[:, :, i].masked_fill_(target_m[:, :, i], 0.0)
        
        # Return combined mask (B, T) - True if ANY channel in group is masked
        return target_m[:, :, idxs].any(dim=2)

    t0 = _t_start("recon_meal_zero")
    mask_meal_region = get_region_mask_and_clear(meal_idxs, x_masked, mask_t) if meal_idxs else None
    _t_end("recon_meal_zero", t0)
    t0 = _t_start("recon_bolus_zero")
    mask_bolus_region = get_region_mask_and_clear(bolus_idxs, x_masked, mask_t) if bolus_idxs else None
    _t_end("recon_bolus_zero", t0)

    # Identify events: (B, T) bool
    meal_events = (x_masked[:, :, meal_idxs] > 0).any(dim=2) if meal_idxs else None
    bolus_events = (x_masked[:, :, bolus_idxs] > 0).any(dim=2) if bolus_idxs else None

    # Helper function to inject events
    # target_mask: (B, T) where we want to fill
    # source_events: (B, T) bool where source events exist
    # target_indices: list of channel indices to write to
    # impulse_val: float
    def inject_events(target_mask_region, source_events, target_indices, impulse_val):
        if target_mask_region is None or source_events is None or not target_indices:
            return

        B, T = target_mask_region.shape
        
        # 1. Direct overlap: If source event is INSIDE the masked region, copy it directly.
        # This handles the case where t maps to t (clamped to itself).
        direct_hits = target_mask_region & source_events
        if direct_hits.any():
             for c in target_indices:
                x_masked[:, :, c].masked_fill_(direct_hits, impulse_val)

        # 2. Boundary logic (Events just outside the mask mapping to start/end of mask)
        # We need to detect starts and ends of masked regions.
        # Start: Mask is True, Previous is False (or t=0)
        # End (exclusive): Mask is False, Previous is True. 
        # But we need the last masked index. 
        # Let's define:
        # region_starts: True at index s where mask[s] is T and mask[s-1] is F
        # region_ends: True at index e where mask[e] is T and mask[e+1] is F (last masked pixel)
        
        # Pad mask to handle edges
        padded_mask = F.pad(target_mask_region, (1, 1), mode='constant', value=False)
        region_starts = padded_mask[:, 1:] & (~padded_mask[:, :-1]) # (B, T)
        region_ends = padded_mask[:, :-2] & (~padded_mask[:, 1:-1]) # (B, T) - this is shifted.
        
        # Actually simplest:
        # Starts: mask[t] & ~mask[t-1]
        # Lasts: mask[t] & ~mask[t+1]
        
        mask_float = target_mask_region.float()
        diff = mask_float[:, 1:] - mask_float[:, :-1]
        
        # starts: diff == 1 (at index t+1). Also check t=0.
        starts_bool = torch.zeros_like(target_mask_region, dtype=torch.bool)
        starts_bool[:, 1:] = (diff == 1)
        starts_bool[:, 0] = target_mask_region[:, 0]
        
        # ends: diff == -1 (at index t). That index is the LAST masked pixel.
        # Also check t=T-1
        lasts_bool = torch.zeros_like(target_mask_region, dtype=torch.bool)
        lasts_bool[:, :-1] = (diff == -1)
        lasts_bool[:, -1] = target_mask_region[:, -1]

        # Check for source events in window [s-k, s-1] for starts
        # We can use convolution to detect "any event in previous k steps"
        source_float = source_events.float().unsqueeze(1) # (B, 1, T)
        kernel = torch.ones((1, 1, k), device=device)
        
        # Convolve. Padding needs to be carefully handled to look "backward" or "forward"
        # For "backward" check (events before current time):
        # At time t, we want sum(source[t-k:t]). 
        # F.conv1d with padding=k, then slice appropriately.
        # Standard conv with kernel k centered? No.
        
        # Let's map explicit windows.
        # For starts at t, check source[t-k : t].
        # Use padded conv.
        # Pad source left by k.
        padded_source = F.pad(source_float, (k, k), mode='constant', value=0.0)
        # Conv result at t corresponds to window ending at t?
        # If we act on padded:
        # We want at index `t` to see sum of `t-k` to `t-1`.
        # This is strictly "past" events.
        
        # Let's just create a "has_recent_event" mask
        # Convolution valid mode
        # Kernel size k.
        # To see [t-k, t-1], we effectively check if `conv(source)` shifted right aligns with `starts`.
        
        # Simpler: Dilate the source events by k
        # But we need directionality.
        
        # Check "Left" (Pre-gap events map to Start)
        # If `source` has event in [t-k, t-1], trigger start at t.
        # This is equivalent to: shift `source` forward by 1..k. If any overlap with `start`, trigger.
        
        for shift in range(1, k + 1):
            # Check source[t - shift]
            # Shift source right by shift
            shifted_source = torch.roll(source_events, shifts=shift, dims=1)
            # Mask out wraparounds (t < shift)
            shifted_source[:, :shift] = False
            
            # If start[t] and source[t-shift], then trigger
            trigger = starts_bool & shifted_source
            if trigger.any():
                for c in target_indices:
                    x_masked[:, :, c].masked_fill_(trigger, impulse_val)
                    
        # Check "Right" (Post-gap events map to End)
        # If `source` has event in [t+1, t+k], trigger end at t.
        # Equivalent to: shift source left by 1..k.
        
        for shift in range(1, k + 1):
            shifted_source = torch.roll(source_events, shifts=-shift, dims=1)
            shifted_source[:, -shift:] = False
            
            trigger = lasts_bool & shifted_source
            if trigger.any():
                for c in target_indices:
                    x_masked[:, :, c].masked_fill_(trigger, impulse_val)

    # 4. Execute injections
    if meal_idxs and mask_meal_region is not None:
        t0 = _t_start("recon_meal_coupling")
        if bolus_events is not None and bool(bolus_events.any()):
            inject_events(mask_meal_region, bolus_events, meal_idxs, meal_impulse)
        _t_end("recon_meal_coupling", t0)
    
    if bolus_idxs and mask_bolus_region is not None:
        t0 = _t_start("recon_bolus_coupling")
        if meal_events is not None and bool(meal_events.any()):
            inject_events(mask_bolus_region, meal_events, bolus_idxs, bolus_impulse)
        _t_end("recon_bolus_coupling", t0)
