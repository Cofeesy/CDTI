import numpy as np
import random
from typing import List
from scipy import optimize
from scipy.special import expit


def pick_coeffs(
    X: np.ndarray,
    idxs_obs: List[int] = [],
    idxs_nas: List[int] = [],
    self_mask: bool = False,
) -> np.ndarray:
    n, d = X.shape
    if self_mask:
        coeffs = np.random.rand(d)
        Wx = X * coeffs
        coeffs /= np.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = np.random.rand(d_obs, d_na)
        Wx = X[:, idxs_obs] @ coeffs
        coeffs /= np.std(Wx, 0, keepdims=True)
    return coeffs


def fit_intercepts(
    X: np.ndarray, coeffs: np.ndarray, p: float, self_mask: bool = False
) -> np.ndarray:
    if self_mask:
        d = len(coeffs)
        intercepts = np.zeros(d)
        for j in range(d):
            def f(x: np.ndarray) -> np.ndarray:
                return expit(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        for j in range(d_na):
            def f(x: np.ndarray) -> np.ndarray:
                return expit(np.dot(X, coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


def get_MCAR_mask(org_data: np.ndarray, missing_p: float, seed: int) -> np.ndarray:
    """
    Generate MCAR (Missing Completely At Random) mask.
    Returns 1 for observed values, 0 for missing ones.
    """
    random.seed(seed)
    np.random.seed(seed)

    mask = np.ones_like(org_data)
    mask[np.where(org_data == -200)] = 0

    n, m = org_data.shape
    missing_target = np.sum(mask) * missing_p
    missing_count = 0

    while missing_count < missing_target:
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        if mask[i, j] == 0:
            continue
        mask[i, j] = 0
        missing_count += 1

    return mask


def get_MAR_mask(org_data: np.ndarray, missing_p: float, seed: int) -> np.ndarray:
    """
    Generate MAR (Missing At Random) mask.
    Returns 1 for observed values, 0 for missing ones.
    """
    random.seed(seed)
    np.random.seed(seed)

    mask = np.ones_like(org_data)
    mask[np.where(org_data == -200)] = 0

    n, m = org_data.shape
    missing_target = np.sum(mask) * missing_p

    attribute_data = org_data[:, 0]
    index = np.argsort(attribute_data)
    rank = np.argsort(index) + 1
    probability = rank / np.sum(rank)

    missing_count = 0
    while missing_count < missing_target:
        attr = random.randint(0, m - 1)
        i = np.random.choice(range(n), p=probability.ravel())
        if mask[i, attr] == 0:
            continue
        mask[i, attr] = 0
        missing_count += 1

    return mask


def get_MNAR_mask(X: np.ndarray, missing_p: float, seed: int, p_params: float = 0.3, exclude_inputs: bool = True) -> np.ndarray:
    """
    Generate MNAR (Missing Not At Random) mask.
    Returns 1 for observed values, 0 for missing ones.
    """
    np.random.seed(seed)
    n, d = X.shape
    mask = np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d
    d_na = d - d_params if exclude_inputs else d

    idxs_params = (
        np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    )
    idxs_nas = (
        np.array([i for i in range(d) if i not in idxs_params])
        if exclude_inputs
        else np.arange(d)
    )

    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, missing_p)

    ps = expit(X[:, idxs_params] @ coeffs + intercepts)
    ber = np.random.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    if exclude_inputs:
        mask[:, idxs_params] = np.random.rand(n, d_params) < missing_p

    mask = (~mask).astype(int)
    return mask


def calc_missing_rate(mask: np.ndarray) -> float:
    """
    Calculate missing rate in mask.
    Args:
        mask: numpy.ndarray, 1 for observed, 0 for missing
    Returns:
        Missing rate (float)
    """
    total = mask.size
    n_missing = np.sum(mask == 0)
    missing_rate = n_missing / total
    return missing_rate


def generate_masks(data: np.ndarray, missing_p: float, seed: int, mechanisms: List[str]) -> dict:
    """
    Generate masks for all specified mechanisms.
    """
    masks = {}
    if 'mcar' in mechanisms:
        masks['mcar'] = get_MCAR_mask(data, missing_p, seed)
    if 'mar' in mechanisms:
        masks['mar'] = get_MAR_mask(data, missing_p, seed)
    if 'mnar' in mechanisms:
        masks['mnar'] = get_MNAR_mask(data, missing_p, seed)
    return masks
