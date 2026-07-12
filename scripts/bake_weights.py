"""Download BLIP weights into the image's HF cache at build time.

Running this in the Dockerfile bakes the ~0.5 GB model into the image so the
container needs no network at runtime and the first ``ingest`` isn't slow.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    model_name = os.environ.get("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
    from transformers import BlipForConditionalGeneration, BlipProcessor

    print(f"Baking weights for {model_name} into {os.environ.get('HF_HOME', '(default)')}")
    BlipProcessor.from_pretrained(model_name)
    BlipForConditionalGeneration.from_pretrained(model_name)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
