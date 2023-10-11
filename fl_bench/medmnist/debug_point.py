from nvflare import SimulatorRunner
import sys
RESULT_ROOT = "/path/to/localscratch/MedMNIST/debug_logs"
algorithms_dir = "/path/to/FL_bench/fl_bench/medmnist/job_configs/medmnist_central"


def run_simulator():
    simulator = SimulatorRunner(
        job_folder=algorithms_dir,
        workspace=RESULT_ROOT,
        n_clients=1,
        threads=1,
        gpu="0",
    )
    run_status = simulator.run()

    return run_status


if __name__ == "__main__":
    status = run_simulator()
    sys.exit(status)
