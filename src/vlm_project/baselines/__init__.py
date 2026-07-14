"""Committed, package-shipped baselines for correctness gates.

Shipping the baseline inside the package (not just the repo) means it travels
into the Docker image and is available to ``verify-accuracy`` wherever the image
runs, no volume mount needed. A rebuilt image carries the last-known-good
numbers, so a dependency/model bump that degrades captions is caught by comparing
the new run against this committed baseline.
"""
