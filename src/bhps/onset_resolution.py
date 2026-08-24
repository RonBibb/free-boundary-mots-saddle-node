"""Pure adjudication helpers for a sampled three-grid formation onset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def persistent_pair_transition(counts: Sequence[int]) -> int | None:
    """Return the first 0->2 transition index, or ``None`` if not persistent."""
    values = [int(value) for value in counts]
    positive = [index for index, value in enumerate(values) if value > 0]
    if not positive:
        return None
    first = positive[0]
    if first == 0:
        return None
    if any(value != 0 for value in values[:first]):
        return None
    if any(value != 2 for value in values[first:]):
        return None
    return first


def onset_summary(
    times: Sequence[float], histories: Mapping[str, Sequence[int]],
    fine_step: float,
) -> dict:
    """Summarize sampled brackets and the sealed localization conditions."""
    sample_times = [float(value) for value in times]
    if not sample_times:
        raise ValueError("at least one sampled time is required")
    if fine_step <= 0.0:
        raise ValueError("fine_step must be positive")
    transitions = {
        label: persistent_pair_transition(counts)
        for label, counts in histories.items()
    }
    complete = bool(transitions and all(value is not None for value in transitions.values()))
    if not complete:
        return {
            "complete": False,
            "transition_indices": transitions,
            "first_detection_times": None,
            "brackets": None,
            "spread": None,
            "spread_below_two_steps": False,
            "G8_G9_lag_not_worse_than_G7_G8_plus_one_step": False,
        }
    first_times = {
        label: sample_times[index] for label, index in transitions.items()
    }
    brackets = {
        label: {
            "lower": sample_times[index - 1],
            "upper": sample_times[index],
            "width": sample_times[index] - sample_times[index - 1],
        }
        for label, index in transitions.items()
    }
    spread = max(first_times.values()) - min(first_times.values())
    lag78 = abs(first_times["G8"] - first_times["G7"])
    lag89 = abs(first_times["G9"] - first_times["G8"])
    epsilon = 32.0 * max(abs(value) for value in sample_times) * 2.220446049250313e-16
    return {
        "complete": True,
        "transition_indices": transitions,
        "first_detection_times": first_times,
        "brackets": brackets,
        "spread": spread,
        "spread_below_two_steps": bool(spread <= 2.0 * fine_step + epsilon),
        "G7_G8_lag": lag78,
        "G8_G9_lag": lag89,
        "G8_G9_lag_not_worse_than_G7_G8_plus_one_step": bool(
            lag89 <= lag78 + fine_step + epsilon
        ),
    }


def endpoint_vector_difference(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float | None:
    """Euclidean difference between two sorted two-branch endpoint vectors."""
    import numpy as np

    a = np.asarray(sorted(left, key=lambda pair: pair[1]), dtype=float).ravel()
    b = np.asarray(sorted(right, key=lambda pair: pair[1]), dtype=float).ravel()
    if a.size != 4 or b.size != 4:
        return None
    return float(np.linalg.norm(a - b))

