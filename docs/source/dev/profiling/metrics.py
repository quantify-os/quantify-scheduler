import cProfile
import pstats
import qcodes
import importlib
import inspect
import numpy as np
import sys
import argparse


# How many times each notebook is run. (The results are averaged.)
SAMPLES = 10
# Experiments to run. They need to be `.py` files (python modules),
# and their `run_experiment` function is profiled.
# Note: for practical purposes, they can be jupytext notebooks.
EXPERIMENT_NOTEBOOKS = [
    "simple_binned_acquisition",
    "resonator_spectroscopy",
    "random_gates",
    "multidim_batched_sweep",
]
# Methods to profile. The function will be profiled in the class.
# (<name>, (<class>, <function name>))
# Note, <name> can be anything, that's what is displayed in the profiling notebook.
METHODS = [
    ("compile", ("QuantifyCompiler", "compile")),
    ("prepare", ("InstrumentCoordinator", "prepare")),
    ("schedule", (None, "create_schedule")),
    ("run", ("InstrumentCoordinator", "start")),
    ("process", ("InstrumentCoordinator", "retrieve_acquisition")),
]


def stat_experiment(experiment_notebook):
    qcodes.instrument.Instrument.close_all()
    if experiment_notebook in sys.modules:
        notebook_mod = importlib.import_module(experiment_notebook)
        importlib.reload(notebook_mod)
    else:
        notebook_mod = importlib.import_module(experiment_notebook)
    with cProfile.Profile() as pr:
        notebook_mod.run_experiment()
    if "close_experiment" in globals():
        close_experiment()
    return pstats.Stats(pr)


def match_class_method(class_name, method_name, module_path, line_number):
    module_name = inspect.getmodulename(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Note, this is an expected behavior.
        # This function runs on *all* modules and functions,
        # which took part in the experiment, and some modules
        # are inherently unloadable.
        # For them, there will be no match, and that is ok.
        return False

    classes = inspect.getmembers(module, inspect.isclass)
    for _, cls in classes:
        methods = inspect.getmembers(cls, inspect.isfunction)
        for method in methods:
            if method[0] == method_name:
                _, start_line = inspect.getsourcelines(method[1])
                if (class_name == cls.__name__) and (start_line == line_number):
                    return True
    return False


def get_stat(stats, class_name, method_name):
    for stat_key in stats.stats:
        module_path = stat_key[0]
        line_number = stat_key[1]
        current_method_name = stat_key[2]
        if method_name == current_method_name:
            if class_name is not None:
                if match_class_method(
                    class_name, method_name, module_path, line_number
                ):
                    return stats.stats[stat_key]
            else:
                return stats.stats[stat_key]
    return None


def expected_value_and_sigma(t_sum, t_sq_sum, samples):
    expected_value = t_sum / samples
    sigma = (t_sq_sum / samples - expected_value**2) ** 0.5
    return (expected_value, sigma)


class Sums:
    def __init__(self, sum=0, sq_sum=0):  # pylint: disable=redefined-builtin
        self.sum = sum
        self.sq_sum = sq_sum


def stat_experiment_detailed(experiment_notebook, samples, methods):
    total_time = Sums()
    times = [Sums() for _ in range(len(methods))]
    for sample in range(samples):
        print(f"Running notebook {experiment_notebook}    {sample + 1}/{samples}")
        stats = stat_experiment(experiment_notebook)

        for i, method in enumerate(methods):
            current_stats = get_stat(stats, method[1][0], method[1][1])
            if current_stats:
                time = current_stats[3]
                times[i].sum += time
                times[i].sq_sum += time**2
            else:
                times[i].sum = np.nan
                times[i].sq_sum = np.nan
        total_time.sum += stats.total_tt
        total_time.sq_sum += stats.total_tt**2

    times = [expected_value_and_sigma(t.sum, t.sq_sum, samples) for t in times]
    total_time = expected_value_and_sigma(total_time.sum, total_time.sq_sum, samples)

    stats.dump_stats(f"{experiment_notebook}.prof")
    print(f"Generated `{experiment_notebook}.prof` profiling file")

    return times, total_time


def measure_experiment_runtimes():
    measured_data = []
    for experiment_notebook in EXPERIMENT_NOTEBOOKS:
        times, total_time = stat_experiment_detailed(
            experiment_notebook, samples=SAMPLES, methods=METHODS
        )
        measured_data.append((experiment_notebook, times, total_time))
    return measured_data


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="Generate profiling metrics for quantify."
    )
    argument_parser.add_argument(
        "filename",
        type=str,
        default="metrics.txt",
        nargs="?",
        help="Output filename.",
    )
    arguments = argument_parser.parse_args()
    with open(file=arguments.filename, mode="w") as file:
        measured_data = measure_experiment_runtimes()
        for experiment_notebook, _, (average, error) in measured_data:
            file.write(
                f'run_time{{experiment="{experiment_notebook}",type="mean"}} {average:.3g}\n'
            )
            file.write(
                f'run_time{{experiment="{experiment_notebook}",type="std_err"}} {error:.1g}\n'
            )
