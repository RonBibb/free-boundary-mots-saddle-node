"""Prospective grading logic for the sealed Test-4C certificate."""


def grade_test4c(
    controls_pass, a790_g9_leaves, a790_g10_leaves,
    a794_unique_root_count, a794_unresolved_count,
    independently_confirmed_a790_root=False,
):
    leaves = list(a790_g9_leaves) + list(a790_g10_leaves)
    validated_a790_root = any(
        item.get("classification") == "validated_root" for item in leaves
    )
    if validated_a790_root and independently_confirmed_a790_root:
        return "FAIL"
    a790_complete_positive = bool(leaves) and all(
        item.get("classification") == "root_free_positive" for item in leaves
    )
    if (
        controls_pass and a790_complete_positive
        and int(a794_unique_root_count) == 2
        and int(a794_unresolved_count) == 0
    ):
        return "PASS"
    return "REVIEW"
