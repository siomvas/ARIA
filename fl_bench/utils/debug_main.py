from nvflare import SimulatorRunner
import sys
import os

RESULT_ROOT = "/path/to/localscratch/JulyTasks/FedIsic/debug"
algorithms_dir = f"{os.getcwd()}/fl_bench/fed_isic/job_configs/fed_isic_central"


def run_simulator():
    simulator = SimulatorRunner(
        job_folder=algorithms_dir,
        workspace=RESULT_ROOT,
        n_clients=1,
        threads=1,
        gpu="1",
    )
    run_status = simulator.run()

    return run_status


if __name__ == "__main__":
    status = run_simulator()
    sys.exit(status)
