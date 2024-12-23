import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from collections import OrderedDict
import numpy as np
import pandas as pd
import subprocess, re, os


def generate_filegroup_dict(ions, targets):
    file_groups = {
        ion: {target: f"{ion}-{target}.csv" for target in targets} for ion in ions
    }
    return file_groups


def load_and_sort_csv(filename, basepath="sputtering-yield-data"):
    """Load and sort a CSV file by energy values."""
    return pd.read_csv(
        os.path.join(basepath, filename), names=["e", "y"], comment="#"
    ).sort_values(by="e")


def load_sputtering_data(file_groups):
    data_dict = {
        group: {key: load_and_sort_csv(file) for key, file in files.items()}
        for group, files in file_groups.items()
    }

    for group in data_dict:
        data_dict[group] = OrderedDict(
            sorted(
                data_dict[group].items(),
                key=lambda x: ["B", "Fe", "Mo", "Cu", "W"].index(x[0]),
            )
        )
    return data_dict


def plot_sputtering_yields(file_groups):
    """
    Plot sputtering yields grouped by ion type.

    Parameters:
        file_groups (dict): Dictionary of ion groups and their associated files.

    Returns:
        None
    """
    data_dict = load_sputtering_data(file_groups)

    styles = {
        "D": ("C1", r"D$\rightarrow$"),
        "He": ("C2", r"He$\rightarrow$"),
        "Ar": ("k", r"Ar$\rightarrow$"),
    }
    line_styles = {"B": "-", "Fe": "--", "Mo": "-.", "Cu": "-x"}

    fig, ax = plt.subplots(figsize=(8, 6))

    group_lines = {}
    for group, datasets in data_dict.items():
        color, label_prefix = styles[group]
        group_lines[group] = []
        for target, df in datasets.items():
            style = line_styles.get(target, "-")
            if group == "Ar" and target == "B":
                target += " (TRIM)"
            (line,) = ax.plot(
                df["e"], df["y"], f"{color}{style}", label=f"{label_prefix}{target}"
            )
            group_lines[group].append(line)
    legend_positions = {
        "D": (0.4, 0.34),
        "He": (0.9, 0.6),
        "Ar": (0.6, 0.85),
    }

    for group, lines in group_lines.items():
        ax.add_artist(
            ax.legend(
                handles=lines,
                title=f"{group}$^+$ ion",
                loc="upper center",
                bbox_to_anchor=legend_positions[group],
            )
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Y (atoms/ion)")
    plt.ylim([1e-4, 10])
    plt.xlim([50, 1e6])


def generate_trim_in(template_path, energy, output_path):
    """Generate TRIM.IN file for a specific energy."""
    with open(template_path, "r") as file:
        data = file.read()
    # Replace placeholder with energy (modify template as needed)
    data = data.replace("ENERGY_PLACEHOLDER", str(energy))
    with open(output_path, "w") as file:
        file.write(data)


def run_trim(trim_path):
    """Run TRIM simulation."""
    subprocess.run([os.path.join(trim_path, "TRIM.exe")], cwd=trim_path)


def extract_sputtering_yield_tdata(file_path):
    """Extract sputtering yield from TDATA.sav file."""
    total_sputtered_atoms = None
    total_incident_ions = None

    with open(file_path, "r") as file:
        for line in file:
            # Extract total sputtered atoms
            if "Total Sputtered Atoms and Energy" in line:
                next_line = next(file).strip()
                total_sputtered_atoms = float(
                    next_line.split()[0].replace(",", ".").replace("E+", "e")
                )

            # Extract total incident ions
            if "ION NUMB:" in line:
                next_line = next(file).strip()
                parts = next_line.split()
                total_incident_ions = int(parts[2].replace(",", ""))

            # Stop if both values are found
            if total_sputtered_atoms is not None and total_incident_ions is not None:
                break

    if total_sputtered_atoms is not None and total_incident_ions is not None:
        sputtering_yield = total_sputtered_atoms / total_incident_ions
        return sputtering_yield
    else:
        raise ValueError("Could not extract required data from TDATA.sav")


def parse_filename(filename):
    """
    Use regular expression to get energy from
    'output_10.txt' or 'output_0.1.txt'
    """

    # match = re.search(r'_(\d+\.\d+)', filename)
    match = re.search(r"_(\d+(?:\.\d+)?)(?=\.)", filename)
    if match:
        return float(match.group(1))

    else:
        raise ValueError(f"Invalid filename format: {filename}")


def parse_output_folder(folder_path):
    """
    Folder with output *.txt files only.
    Now parsing sputtering yields
    """
    ls = os.listdir(folder_path)
    return pd.DataFrame(
        [
            (
                parse_filename(i) * 1000,
                extract_sputtering_yield_tdata(f"TRIM_RESULTS/{i}"),
            )
            for i in ls
        ],
        columns=["e", "y"],
    )


def generate_scaled_sequence(start, stop):
    """
    Generate a sequence with increments that scale by order of magnitude.

    Parameters:
    - start: Starting value of the sequence (e.g., 10).
    - stop: Final value of the sequence (e.g., 1e6).

    Returns:
    - A list containing the generated sequence.
    """
    result = []
    current = start
    stop = int(stop)  # Ensure stop is an integer
    while current <= stop:
        step = 10 ** (
            len(str(current)) - 1
        )  # Determine step size by order of magnitude
        result.extend(range(current, min(current * 10, stop + 1), step))
        current *= 10
    return result


def data_dict_to_map(data_dict, energy):
    ions = list(data_dict.keys())
    targets = list(data_dict[ions[0]].keys())
    sputtering_map = pd.DataFrame(index=ions, columns=targets)

    for ion, targets_data in data_dict.items():
        for target, df in targets_data.items():
            # Find the closest energy level
            closest_row = df.iloc[(df["e"] - energy).abs().argmin()]
            sputtering_map.loc[ion, target] = closest_row["y"]

    return sputtering_map.astype(float), ions, targets


def plot_sputtering_map(data_dict, energy):
    sputtering_map, ions, targets = data_dict_to_map(data_dict, energy)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        sputtering_map,
        cmap="cividis",
        aspect="auto",
        interpolation="nearest",
        norm=LogNorm(),
    )

    # Add axis labels and ticks
    plt.xticks(ticks=range(len(targets)), labels=targets)
    plt.yticks(ticks=range(len(ions)), labels=ions)
    plt.xlabel("Target Atom")
    plt.ylabel("Ion")
    plt.title(f"Sputtering at {energy} eV (Log Scale)")

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label("Sputtering Yield")


def barplot2_sputtering(data_dict, energy):
    """"""
    sputtering_map, ions, targets = data_dict_to_map(data_dict, energy)
    colors = ["#ffbb3d", "#9edb42", "#7cb0c4", "#d17fe3"]
    hatches = ["", "\\", "/", "x"]  #'x'

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(targets))  # X-axis positions for target atoms
    bar_width = 0.25  # Width of each bar

    [plt.axhline(j, zorder=0, color="k", lw=0.3) for j in [1e-3, 1e-2, 0.1, 1]]

    for i, (ion, color, hatch) in enumerate(zip(ions, colors, hatches)):
        ax.bar(
            x + i * bar_width,
            sputtering_map.loc[ion],
            bar_width,
            label=f"Ion: {ion}",
            color=color,
            edgecolor="black",
            hatch=hatch,
        )
        for pos, value in zip(x + i * bar_width, sputtering_map.loc[ion]):
            ax.text(
                pos,
                value * 0.5,  # Slightly above the bar
                f"{ion}",
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    alpha=0.95,
                    edgecolor="none",
                    boxstyle="round,pad=0.3",
                ),
            )

    ax.set_xlabel("Target Atom", fontsize=16)  # Larger X-axis label font
    ax.set_ylabel("Y (atoms/ions)", fontsize=16)  # Larger Y-axis label font
    ax.set_title(f"{energy} eV", fontsize=18)  # Larger title font
    ax.set_xticks(x + bar_width * (len(ions) - 1) / 2)
    ax.set_xticklabels(targets, fontsize=14)  # Larger tick label font
    ax.tick_params(axis="both", which="major", labelsize=14)  # Larger tick labels
    ax.set_yscale("log")
    ax.set_ylim([1e-4, 10])

    handles = [
        Patch(
            facecolor=color,
            edgecolor="black",
            hatch=hatch,
            label=f"$\\mathrm{{{ion}^{{+}}}}$",
        )
        for ion, color, hatch in zip(ions, colors, hatches)
    ]
    ax.legend(handles=handles, loc=2, fontsize=14)
