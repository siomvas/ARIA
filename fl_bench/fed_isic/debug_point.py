from nvflare import SimulatorRunner
import sys
RESULT_ROOT = "/path/to/localscratch/Fed_ISIC/debug"
algorithms_dir = "/path/to/FL_bench/fl_bench/fed_isic/job_configs/fed_isic_federated_fedinit"


def run_simulator():
    simulator = SimulatorRunner(
        job_folder=algorithms_dir,
        workspace=RESULT_ROOT,
        n_clients=6,
        gpu="1,1,1,3,3,3",
    )
    run_status = simulator.run()

    return run_status


if __name__ == "__main__":
    status = run_simulator()
    sys.exit(status)