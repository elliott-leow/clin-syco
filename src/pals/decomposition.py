"""Direction decomposition: project, residualize, and analyze direction relationships."""

import torch
import torch.nn.functional as F
import numpy as np


def project_direction(v, onto):
    """Project direction v onto direction 'onto'. Returns the component of v along onto."""
    dot = (v @ onto).item()
    return dot * onto


def residualize(v, onto):
    """Remove the component of v along 'onto'. Returns the orthogonal residual."""
    proj = project_direction(v, onto)
    return v - proj


def decompose_direction(target, components):
    """Decompose a target direction into components along given directions + residual.

    Args:
        target: (hidden_dim,) tensor — the direction to decompose.
        components: dict mapping name -> (hidden_dim,) tensor.

    Returns:
        dict with:
          - "projections": {name: scalar} — projection magnitude along each component
          - "variance_explained": {name: float} — fraction of variance explained
          - "residual_norm": float — norm of the residual after removing all components
          - "total_norm": float — norm of the target
    """
    result = {"projections": {}, "variance_explained": {}, "unique_variance_explained": {}}
    total_var = target.norm().item() ** 2
    residual = target.clone()

    # Marginal projections (for reporting)
    for name, comp in components.items():
        comp_normed = F.normalize(comp, dim=0)
        proj_scalar = (target @ comp_normed).item()
        result["projections"][name] = proj_scalar
        result["variance_explained"][name] = proj_scalar ** 2 / total_var if total_var > 0 else 0.0

    # Sequential residualization (Gram-Schmidt style) with unique variance at each step
    for name, comp in components.items():
        comp_normed = F.normalize(comp, dim=0)
        before_var = residual.norm().item() ** 2
        proj = (residual @ comp_normed) * comp_normed
        residual = residual - proj
        after_var = residual.norm().item() ** 2
        result["unique_variance_explained"][name] = (before_var - after_var) / total_var if total_var > 0 else 0.0

    result["residual_norm"] = residual.norm().item()
    result["total_norm"] = target.norm().item()
    result["residual_variance_fraction"] = (residual.norm().item() ** 2) / total_var if total_var > 0 else 0.0

    return result


def decompose_by_layer(target_dirs, component_dirs_dict):
    """Decompose target directions at each layer.

    Args:
        target_dirs: dict[int, Tensor] — layer -> direction.
        component_dirs_dict: dict[str, dict[int, Tensor]] — name -> {layer -> direction}.

    Returns:
        dict[int, dict] — layer -> decomposition result.
    """
    layers = sorted(target_dirs.keys())
    results = {}
    for layer in layers:
        components = {name: dirs[layer] for name, dirs in component_dirs_dict.items()
                      if layer in dirs}
        results[layer] = decompose_direction(target_dirs[layer], components)
    return results


def decompose_ols(target, components):
    """Order-invariant decomposition using OLS regression.

    Simultaneously regresses target onto all component directions.
    Unlike Gram-Schmidt, the result does not depend on component ordering.

    Args:
        target: (hidden_dim,) tensor.
        components: dict mapping name -> (hidden_dim,) tensor.

    Returns:
        dict with:
          - "coefficients": {name: float} — OLS regression coefficients
          - "variance_explained": {name: float} — R² contribution per component
          - "total_r2": float — total R² of the regression
          - "residual_variance_fraction": float — 1 - total_r2
    """
    names = list(components.keys())
    if not names:
        return {"coefficients": {}, "variance_explained": {},
                "total_r2": 0.0, "residual_variance_fraction": 1.0}

    # Build design matrix: each column is a normalized component direction
    X = torch.stack([F.normalize(components[n], dim=0) for n in names], dim=1)  # (hidden_dim, k)
    y = target  # (hidden_dim,)

    # OLS: beta = (X^T X)^{-1} X^T y
    XtX = X.T @ X  # (k, k)
    Xty = X.T @ y  # (k,)

    # Use pseudoinverse for numerical stability
    beta = torch.linalg.lstsq(XtX, Xty).solution  # (k,)

    # Predicted and residual
    y_hat = X @ beta
    total_var = (y @ y).item()
    explained_var = (y_hat @ y_hat).item()
    total_r2 = explained_var / total_var if total_var > 0 else 0.0

    # Per-component R² contribution via Type III (unique) sum of squares
    # For each component, compare full model R² vs model without that component
    result = {"coefficients": {}, "variance_explained": {}}
    for i, name in enumerate(names):
        result["coefficients"][name] = float(beta[i].item())

        # Model without component i
        mask = [j for j in range(len(names)) if j != i]
        if mask:
            X_reduced = X[:, mask]
            XtX_r = X_reduced.T @ X_reduced
            Xty_r = X_reduced.T @ y
            beta_r = torch.linalg.lstsq(XtX_r, Xty_r).solution
            y_hat_r = X_reduced @ beta_r
            r2_reduced = (y_hat_r @ y_hat_r).item() / total_var if total_var > 0 else 0.0
        else:
            r2_reduced = 0.0

        result["variance_explained"][name] = max(0.0, total_r2 - r2_reduced)

    result["total_r2"] = float(total_r2)
    result["residual_variance_fraction"] = float(max(0.0, 1.0 - total_r2))

    return result


def decompose_ols_by_layer(target_dirs, component_dirs_dict):
    """OLS decomposition at each layer. Order-invariant alternative to decompose_by_layer."""
    layers = sorted(target_dirs.keys())
    results = {}
    for layer in layers:
        components = {name: dirs[layer] for name, dirs in component_dirs_dict.items()
                      if layer in dirs}
        results[layer] = decompose_ols(target_dirs[layer], components)
    return results


def pairwise_cosine_matrix(directions_dict):
    """Compute pairwise cosine similarity matrix between named directions.

    Args:
        directions_dict: dict mapping name -> (hidden_dim,) tensor.

    Returns:
        (names_list, matrix) where matrix[i,j] is cos(dir_i, dir_j).
    """
    names = sorted(directions_dict.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            cos = F.cosine_similarity(
                directions_dict[name_i].unsqueeze(0),
                directions_dict[name_j].unsqueeze(0)
            ).item()
            matrix[i, j] = cos

    return names, matrix
