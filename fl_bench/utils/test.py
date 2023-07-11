import json
from fl_bench.fed_isic.learners.fed_isic_base_learner import FedIsicLearner


def main():
    fed = FedIsicLearner(
            "/path/to/FL_bench/fl_bench/fed_isic/job_configs/fed_isic_fedavg/fed_isic_fedavg/config_train.json",
            central=True
        )
    fed.train_config("")


if __name__ == "__main__":
    main()