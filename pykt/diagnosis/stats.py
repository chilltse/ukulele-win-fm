import math


def new_stat():
    return {
        "n": 0,
        "sum_true": 0.0,
        "sum_true2": 0.0,
        "sum_pred": 0.0,
        "sum_pred2": 0.0,
        "sum_res": 0.0,
        "sum_abs_res": 0.0,
        "sum_sq_res": 0.0,
        "sum_res2": 0.0,
    }


def update_stat(st, true_value, pred_value, residual):
    t = float(true_value)
    p = float(pred_value)
    r = float(residual)

    st["n"] += 1
    st["sum_true"] += t
    st["sum_true2"] += t * t
    st["sum_pred"] += p
    st["sum_pred2"] += p * p
    st["sum_res"] += r
    st["sum_abs_res"] += abs(r)
    st["sum_sq_res"] += r * r
    st["sum_res2"] += r * r


def finalize_stat(st):
    n = st["n"]
    if n <= 0:
        return {
            "n": 0,
            "mean_true": float("nan"),
            "mean_pred": float("nan"),
            "mean_signed_residual": float("nan"),
            "mean_abs_residual": float("nan"),
            "rmse": float("nan"),
            "residual_var": float("nan"),
            "residual_std": float("nan"),
            "residual_se": float("nan"),
            "residual_ci95_low": float("nan"),
            "residual_ci95_high": float("nan"),
            "pred_var": float("nan"),
            "pred_std": float("nan"),
            "true_var": float("nan"),
            "true_std": float("nan"),
        }

    mean_true = st["sum_true"] / n
    mean_pred = st["sum_pred"] / n
    mean_res = st["sum_res"] / n
    mean_abs_res = st["sum_abs_res"] / n
    rmse = math.sqrt(st["sum_sq_res"] / n)

    residual_var = max(st["sum_res2"] / n - mean_res * mean_res, 0.0)
    residual_std = math.sqrt(residual_var)
    residual_se = residual_std / math.sqrt(n) if n > 1 else float("nan")

    pred_var = max(st["sum_pred2"] / n - mean_pred * mean_pred, 0.0)
    true_var = max(st["sum_true2"] / n - mean_true * mean_true, 0.0)

    if math.isnan(residual_se):
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        ci_low = mean_res - 1.96 * residual_se
        ci_high = mean_res + 1.96 * residual_se

    return {
        "n": n,
        "mean_true": mean_true,
        "mean_pred": mean_pred,
        "mean_signed_residual": mean_res,
        "mean_abs_residual": mean_abs_res,
        "rmse": rmse,
        "residual_var": residual_var,
        "residual_std": residual_std,
        "residual_se": residual_se,
        "residual_ci95_low": ci_low,
        "residual_ci95_high": ci_high,
        "pred_var": pred_var,
        "pred_std": math.sqrt(pred_var),
        "true_var": true_var,
        "true_std": math.sqrt(true_var),
    }


def weighted_mean(values, weights):
    if len(values) == 0:
        return float("nan")
    sw = sum(weights)
    if sw <= 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights)) / sw


def weighted_var(values, weights):
    if len(values) == 0:
        return float("nan")
    sw = sum(weights)
    if sw <= 0:
        return float("nan")
    m = weighted_mean(values, weights)
    return sum(w * (v - m) ** 2 for v, w in zip(values, weights)) / sw
