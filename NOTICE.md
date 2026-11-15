# Third-party code

## NVIDIA FLARE

Apache License 2.0, Copyright (c) NVIDIA CORPORATION. https://github.com/NVIDIA/NVFlare

- `fl_bench/utils/fedopt.py` is `nvflare/app_opt/pt/fedopt.py` of NVFlare 2.4.0, unchanged.
- The learners, trainers, validators, formatters and job configurations under `fl_bench` started from NVFlare's simulator examples and were modified for this work. Files that carry the NVIDIA copyright header keep it.
- `patches/nvflare-2.3.0-timeouts.patch` is a patch against NVFlare 2.3.0.

The text of the license is at https://www.apache.org/licenses/LICENSE-2.0.
