# ARIA

Code for **ARIA: On the Interaction Between Architectures, Initialization and Aggregation Methods for Federated Visual Classification** (IEEE ISBI 2024).

[Paper (arXiv)](https://arxiv.org/abs/2311.14625)

ARIA is a joint study of the three choices every federated image classifier has to make: the architecture, how it is initialized and how client models are aggregated. It benchmarks their combinations on medical image classification and shows that they have to be chosen together. This repository holds the benchmark: the federated training and evaluation code, the job configurations, the data partitions and the scripts that ran the grid.

## What is here

| Path | Content |
|---|---|
| `fl_bench/fed_isic` | Fed-ISIC2019: learner, trainers, validator, job configurations and the scripts of the grid |
| `fl_bench/medmnist` | OrganAMNIST: the same, with the Dirichlet partitions (alpha 0.1 and 100) under `splits` |
| `fl_bench/networks` | Model factories. One function per architecture and initialization (random, ImageNet, self-supervised) |
| `fl_bench/ssl_pretraining` | DINO pre-training on unlabeled images of each task, which gives the self-supervised initializations |
| `fl_bench/utils` | Shared helpers, SCAFFOLD support and the FedOpt generator |
| `fl_bench/extra` | Batch scripts for the additional OrganAMNIST runs |
| `patches` | An optional patch that raises NVFlare's simulator time-outs |

Federated training runs in the [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) simulator. Aggregation methods: FedAvg, FedOpt, FedProx and SCAFFOLD.

## Setup

The experiments ran with Python 3.8, PyTorch 2.1 and NVFlare 2.3.0.

```bash
git clone https://github.com/siomvas/ARIA.git && cd ARIA
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

FedOpt needs no extra step. The experiments used the FedOpt generator that NVFlare released in 2.4.0, which handles models with batch normalization correctly, and that file ships here as `fl_bench/utils/fedopt.py`.

If clients time out during long local training, apply the time-out patch to your NVFlare install:

```bash
cd $(python -c "import nvflare, os; print(os.path.dirname(os.path.dirname(nvflare.__file__)))")
patch -p1 < /path/to/ARIA/patches/nvflare-2.3.0-timeouts.patch
```

## Data and weights

Copy `.env.example` to `.env` and set the folders.

| Variable | What it points to |
|---|---|
| `FEDISIC_ROOT` | Fed-ISIC2019, preprocessed as in [FLamby](https://github.com/owkin/FLamby), with its `train_test_split` file |
| `MEDMNIST_ROOT` | The folder where the `medmnist` package keeps OrganAMNIST |
| `ARIA_WEIGHTS_DIR` | Downloaded initializations that are not fetched automatically (MoCo v3) |
| `ARIA_SSL_FED_ISIC_DIR`, `ARIA_SSL_ORGANAMNIST_DIR` | The DINO backbones produced by `fl_bench/ssl_pretraining` |

ImageNet weights come from timm and torchvision on first use.

## Running

Each run has two steps. `setup.py` writes the choices into a job configuration, and the NVFlare simulator executes it.

```bash
cd fl_bench/medmnist
python src/setup.py -job job_configs/medmnist_federated -method fedopt -alpha 100 \
    -model_name swin_t -num_rounds 50 -local_steps 200 -lr 0.01 --batch_size 64 -seed 111 --amp \
    --opt_args '{"class": "SGD", "momentum": 0.9, "nesterov": true, "weight_decay": 1e-5}'
nvflare simulator job_configs/medmnist_federated -w /path/to/results/run_name -n 4 -gpu 0,0,0,0
```

The shell scripts in `fl_bench/fed_isic/scripts`, `fl_bench/medmnist/scripts` and `fl_bench/extra` are the ones that produced the paper's tables. They are kept as they ran, with `/path/to` in place of the folders of our cluster.

Self-supervised pre-training:

```bash
python fl_bench/ssl_pretraining/dino_fed_isic.py --model_name resnet_50 --data_dir /path/to/unlabeled/images
```

## Citation

```bibtex
@inproceedings{siomos2024aria,
  title        = {ARIA: On the Interaction Between Architectures, Initialization and Aggregation Methods for Federated Visual Classification},
  author       = {Siomos, Vasilis and Naval-Marimont, Sergio and Passerat-Palmbach, Jonathan and Tarroni, Giacomo},
  booktitle    = {2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  pages        = {1--5},
  year         = {2024},
  organization = {IEEE}
}
```

## License

MIT, see `LICENSE.md`. Parts of the training code derive from NVIDIA FLARE examples and `fl_bench/utils/fedopt.py` is an NVIDIA FLARE file. Both are under the Apache License 2.0, see `NOTICE.md`.

Follow-up work: [ANFR](https://github.com/siomvas/ANFR), [FedCLAM](https://github.com/siomvas/FedCLAM) and [FDS](https://github.com/siomvas/fds).
