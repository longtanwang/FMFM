import numpy as np

def maxpmi_optimized(ampbi: np.ndarray) -> tuple:
    """
    Calculates the optimal split point(s) in a binary array to maximize a
    normalized pointwise mutual information (NPMI) like score.

    Args:
        ampbi (np.ndarray): A 1D NumPy array of binary values (0s and 1s).

    Returns:
        tuple: A tuple containing three elements:
               - np.ndarray: The mutual information (MI) values for the best cut(s).
               - float: The maximum score (enpmivalue) found.
               - np.ndarray: An array of the optimal cut location(s) (indices).
               Returns (-1, -1, empty_array) if no valid split is possible.
    """
    # --- 1. Initial Sanity Checks and Setup ---
    n = len(ampbi)
    # Handle arrays that are too short to be split.
    if n < 2:
        return -1, -1, np.array([], dtype=int)
    zeronum_total = np.count_nonzero(ampbi == 0)
    onenum_total = n - zeronum_total
    if zeronum_total == 0 or onenum_total == 0:
        return -1, -1, np.array([], dtype=int)

    # --- 2. Identify all potential split points ---
    t_num = np.where(ampbi[:-1] < ampbi[1:])[0] + 1
    if len(t_num) == 0:
        return -1, -1, np.array([], dtype=int)

    # --- 3. Vectorized Counting of 0s and 1s for all Partitions ---
    is_zero = (ampbi == 0).astype(np.int32)
    cumsum_zeros = np.cumsum(is_zero)

    # For each potential split point `t` in `t_num`, calculate the counts
    # in the left partition (0 to t-1) and right partition (t to n).
    zeronum0 = cumsum_zeros[t_num - 1]
    onenum0 = t_num - zeronum0
    zeronum1 = zeronum_total - zeronum0
    onenum1 = onenum_total - onenum0

    # --- 4. Vectorized PMI and NPMI Calculation ---
    # Calculate joint and marginal probabilities.
    p00 = zeronum0 / n; p01 = zeronum1 / n; p10 = onenum0 / n; p11 = onenum1 / n
    p_x0 = zeronum_total / n; p_x1 = onenum_total / n; p_y0 = t_num / n; p_y1 = (n - t_num) / n

    pmi_val = np.zeros((len(t_num), 4))
    npmi_val = np.zeros((len(t_num), 4))

    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate terms for the contingency table [0,0], [0,1], [1,0], [1,1]
        mask = (p00 > 0); pmi_val[mask, 0] = p00[mask] * np.log2(p00[mask] / (p_x0 * p_y0[mask])); npmi_val[mask, 0] = (-1) * np.log2(p00[mask] / (p_x0 * p_y0[mask])) / np.log2(p00[mask])
        mask = (p01 > 0); pmi_val[mask, 1] = p01[mask] * np.log2(p01[mask] / (p_x0 * p_y1[mask])); npmi_val[mask, 1] = (-1) * np.log2(p01[mask] / (p_x0 * p_y1[mask])) / np.log2(p01[mask])
        mask = (p10 > 0); pmi_val[mask, 2] = p10[mask] * np.log2(p10[mask] / (p_x1 * p_y0[mask])); npmi_val[mask, 2] = (-1) * np.log2(p10[mask] / (p_x1 * p_y0[mask])) / np.log2(p10[mask])
        mask = (p11 > 0); pmi_val[mask, 3] = p11[mask] * np.log2(p11[mask] / (p_x1 * p_y1[mask])); npmi_val[mask, 3] = (-1) * np.log2(p11[mask] / (p_x1 * p_y1[mask])) / np.log2(p11[mask])
    
    # Clean up any resulting NaN values from the calculations.
    pmi_val = np.nan_to_num(pmi_val)
    npmi_val = np.nan_to_num(npmi_val, nan=-1.0)

    # --- 5. Determine the Best Split(s) ---
    mivalue = np.sum(pmi_val, axis=1)
    enpmivalue = (npmi_val[:, 0] * p00) - (npmi_val[:, 1] * p01) - (npmi_val[:, 2] * p10) + (npmi_val[:, 3] * p11)
    max_enpmivalue = np.max(enpmivalue)
    best_indices = np.where(enpmivalue == max_enpmivalue)[0]
    return mivalue[best_indices], max_enpmivalue, t_num[best_indices]