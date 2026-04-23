"""Deprecated helper kept only for backward compatibility.

This project no longer uses exact_feature.py in the final pipeline.
Use prepare_data.py, train_layer1.py, train_layer2.py, benchmark_models.py,
and eval_cascade.py instead.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        'exact_feature.py is deprecated and not used in the final pipeline. '
        'Please use the scripts under project/malware_pipeline/scripts instead.'
    )


if __name__ == '__main__':
    main()
