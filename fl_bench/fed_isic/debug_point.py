from nvflare import SimulatorRunner
import sys
RESULT_ROOT = "/path/to/localscratch/Fed_ISIC/debug"
algorithms_dir = "/path/to/FL_bench/fl_bench/fed_isic/job_configs/fed_isic_federated"


def run_simulator():
    simulator = SimulatorRunner(
        job_folder=algorithms_dir,
        workspace=RESULT_ROOT,
        n_clients=6,
        gpu="0,0,0,1,1,1",
    )
    run_status = simulator.run()

    return run_status


if __name__ == "__main__":
    status = run_simulator()
    sys.exit(status)