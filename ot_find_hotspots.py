#!/usr/bin/env python3
# ot_find_hotspots.py -- scans a gain reference file for isolated hot-pixel calibration defects
# Created 20260901 (David Jurkowitz - Gan Lab using Claude)
# Revised 20260901 17:38
#
# -----------------------------------------------------------------------------
# Generated with the assistance of Claude (Anthropic).
# Model: Claude Sonnet 5 (claude-sonnet-5)        Date: 2026-09-01
# Reviewed, tested, and validated by the Gan Lab (Anaphase); the authors take
# full responsibility for the correctness of this code.
# Lab: https://github.com/anaphaze | https://www.anaphase.org
# -----------------------------------------------------------------------------
"""
ot_find_hotspots.py

Scans an EER gain reference (.mrc) for hot-pixel defects: pixels whose value
deviates from the local median so far beyond the rest of the array's spread
that they represent a genuine, isolated hardware/calibration defect rather
than normal flat-field variation or the known edge/corner effect at the
sensor border.

METHOD
  1. Exclude a border margin (default 5 px) on all sides before scanning.
     Sensor-edge pixels are a separate, already-characterized phenomenon
     and would otherwise dominate any top-N or threshold-based scan.
  2. Sort the remaining (interior) pixels by |value - median| descending.
  3. Walk down that sorted list looking for a large multiplicative jump
     between consecutive deviations (an "elbow"). Everything at or above
     the elbow is reported as a hotspot; everything below it is treated
     as the normal tail of the distribution.

This is deliberately not a fixed formula tied to the single largest value
(e.g. median + max/2) -- it looks at the *shape* of the sorted-deviation
curve, so it will still find multiple independent defects of differing
magnitude, as long as each stands out from whatever comes after it.

DEPENDENCIES
  numpy, mrcfile (required); matplotlib (optional, only for -plot).
  If missing, running this script will print install instructions for
  Linux and macOS, both user-only and system-wide.

SCRIPT IS NON-INTERACTIVE. Run with no arguments to see the full help
menu and a sample input-parameter file.

TWO-SCRIPT PIPELINE
  This script's hotspot coordinates are written to <basename>_hotspots.meta,
  which ot_patch_gain_reference.py reads automatically (via -metadata, or by
  auto-derived filename) so the coordinates don't need to be retyped by hand.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Dependency check -- print install instructions rather than a bare traceback
# ---------------------------------------------------------------------------
_MISSING = []
try:
    import numpy as np
except ImportError:
    _MISSING.append("numpy")
try:
    import mrcfile
except ImportError:
    _MISSING.append("mrcfile")

if _MISSING:
    pkgs = " ".join(_MISSING)
    print(f"Missing required Python module(s): {pkgs}")
    print()
    print("Install instructions:")
    print()
    print("  Linux / macOS, user-only install (no admin rights needed):")
    print(f"    pip install --user {pkgs}")
    print()
    print("  Linux / macOS, system-wide install:")
    print(f"    sudo pip install {pkgs}")
    print()
    print("  If your system uses an externally-managed Python environment")
    print("  (common on newer Linux distros / Homebrew Python on macOS):")
    print(f"    pip install --break-system-packages {pkgs}")
    sys.exit(1)

_HAVE_MATPLOTLIB = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    _HAVE_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Option definitions: (canonical_name, type, default, group, help_text)
#
# Groups per lab convention: BIOPHYSICAL, ALGORITHM/PARALLELIZATION, GRAPHICAL.
# BIOPHYSICAL is intentionally empty -- this script operates on a detector
# calibration file, not a biophysical/sample model. Listed here only to
# document that it was considered, not omitted by oversight.
# ---------------------------------------------------------------------------
OPTIONS = [
    ("border",    int,   5,    "ALGORITHM", "Pixels excluded from each edge before scanning (sensor-edge effects are a separate, known phenomenon)."),
    ("gap_ratio", float, 3.0,  "ALGORITHM", "Ratio between consecutive sorted deviations that defines a genuine outlier 'elbow'."),
    ("n_show",    int,   10,   "ALGORITHM", "How many top interior deviations to report regardless of the elbow."),
    ("input",     str,   None, "ALGORITHM", "Path to an input-parameter file (key=value per line) to load settings from."),
    ("output",    str,   None, "ALGORITHM", "Basename override for all output files (parameter file, log, diagnostic plot). Default: derived from the gain file name."),
    ("defaults",  bool,  False,"ALGORITHM", "Ignore any parameter file / other flags above and use the built-in defaults."),
    ("plot",      bool,  False,"GRAPHICAL", "Save a diagnostic PNG of the gain reference with detected hotspot(s) circled."),
]
_NAMES = [o[0] for o in OPTIONS]
_TYPES = {o[0]: o[1] for o in OPTIONS}
_DEFAULTS = {o[0]: o[2] for o in OPTIONS}


# ---------------------------------------------------------------------------
# Minimal-unique-prefix single-dash option parsing
# ---------------------------------------------------------------------------
def resolve_option(token, names):
    """Return the canonical option name matching a (possibly abbreviated)
    token. Raises ValueError with a human-readable message if the token
    matches zero or more than one option."""
    candidates = [n for n in names if n.startswith(token)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise ValueError(f"unknown option '-{token}'. Run with no arguments or -h for the full option list.")
    matches = ", ".join("-" + c for c in candidates)
    raise ValueError(
        f"ambiguous option '-{token}': could match {matches}. "
        f"Type more characters to uniquely identify the one you want."
    )


def parse_args(argv):
    """Parse argv (sys.argv[1:]). Returns one of:
        ("HELP", None, None)
        (gain_path, values_dict, explicit_set)
    explicit_set records which canonical names were set on the command
    line, so a later -input file only fills in what wasn't explicitly given.
    """
    values = dict(_DEFAULTS)
    explicit = set()
    gain_path = None

    i = 0
    while i < len(argv):
        tok = argv[i]

        if tok in ("-h", "--help"):
            return "HELP", None, None

        if tok.startswith("-") and tok != "-":
            body = tok.lstrip("-")
            if "=" in body:
                body, inline_val = body.split("=", 1)
            else:
                inline_val = None

            try:
                canon = resolve_option(body, _NAMES)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(2)

            if _TYPES[canon] is bool:
                values[canon] = True
                explicit.add(canon)
                i += 1
                continue

            if inline_val is not None:
                raw = inline_val
                i += 1
            else:
                if i + 1 >= len(argv):
                    print(f"Error: -{canon} requires a value.")
                    sys.exit(2)
                raw = argv[i + 1]
                i += 2

            try:
                values[canon] = _TYPES[canon](raw)
            except ValueError:
                print(f"Error: -{canon} expects a {_TYPES[canon].__name__}, got '{raw}'.")
                sys.exit(2)
            explicit.add(canon)
            continue

        # positional argument (the gain file path)
        if gain_path is None:
            gain_path = tok
            i += 1
        else:
            print(f"Error: unexpected extra argument '{tok}'.")
            sys.exit(2)

    return gain_path, values, explicit


def load_param_file(path):
    """Parse a simple key=value input-parameter file. Unknown/ambiguous
    keys are warned about (not fatal) so a stray typo doesn't kill a run
    that's otherwise fine."""
    loaded = {}
    with open(path) as fh:
        for line_num, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                print(f"Warning: ignoring malformed line {line_num} in {path}: '{line}'")
                continue
            key, val = (s.strip() for s in line.split("=", 1))
            if val == "":
                continue
            try:
                canon = resolve_option(key, _NAMES)
            except ValueError as e:
                print(f"Warning in parameter file {path}, line {line_num}: {e}")
                continue
            if _TYPES[canon] is bool:
                loaded[canon] = val.lower() in ("1", "true", "yes")
            else:
                try:
                    loaded[canon] = _TYPES[canon](val)
                except ValueError:
                    print(f"Warning in parameter file {path}, line {line_num}: "
                          f"'{val}' is not a valid {_TYPES[canon].__name__} for -{canon}, ignoring.")
    return loaded


# ---------------------------------------------------------------------------
# Help menu / sample parameter file
# ---------------------------------------------------------------------------
def print_help():
    print(__doc__.strip())
    print()
    print("USAGE")
    print("  ot_find_hotspots.py GAIN_FILE.mrc [options]")
    print()
    for group in ("BIOPHYSICAL", "ALGORITHM/PARALLELIZATION", "GRAPHICAL"):
        print(group)
        members = [o for o in OPTIONS if o[3] == group or (group.startswith(o[3]))]
        if not members:
            print("  (none -- this script has no biophysical/sample-model parameters)")
        for name, typ, default, _grp, help_text in members:
            flag = f"-{name}"
            defstr = "off" if typ is bool else str(default)
            print(f"  {flag:<12} (default: {defstr})  {help_text}")
        if group == "ALGORITHM/PARALLELIZATION":
            print("  Parallelization note: the scan itself is already fully vectorized")
            print("  via NumPy (single process, sub-second for a 4096x4096 array). If")
            print("  you need to scan many gain-reference files, parallelize ACROSS")
            print("  FILES rather than pixels, e.g.:")
            print("    ls *.mrc | xargs -P4 -I{} python3 ot_find_hotspots.py {} -d")
        print()
    print("  -h / --help    Show this help menu.")
    print()
    print("INPUT-PARAMETER FILE FORMAT")
    print("  Plain text, one 'key=value' pair per line, '#' for comments.")
    print("  Keys accept the same minimal-unique abbreviations as the command")
    print("  line. Load one with -input <file>.")
    print()
    print("EXAMPLE (shortest command that reproduces the all-defaults run):")
    print("  ot_find_hotspots.py GainReference.mrc -d")


def write_sample_param_file(basename):
    path = f"{basename}_defaults.param"
    with open(path, "w") as fh:
        fh.write("# Sample input-parameter file for ot_find_hotspots.py, auto-generated.\n")
        fh.write("# Edit values as needed, then run:\n")
        fh.write(f"#   ot_find_hotspots.py GainReference.mrc -input {path}\n")
        for name, typ, default, group, _help in OPTIONS:
            if name in ("input", "defaults"):
                continue  # meta-options, not analysis parameters
            if typ is bool:
                fh.write(f"{name}={'true' if default else 'false'}\n")
            else:
                fh.write(f"{name}={'' if default is None else default}\n")
    print(f"Wrote sample parameter file: {path}")
    return path


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------
def find_hotspots(gain, border, gap_ratio):
    h, w = gain.shape
    interior = gain[border:h - border, border:w - border]
    median = np.median(interior)

    dev = np.abs(interior - median)
    flat_idx_sorted = np.argsort(dev.ravel())[::-1]
    dev_sorted = dev.ravel()[flat_idx_sorted]

    elbow = None
    for i in range(len(dev_sorted) - 1):
        if dev_sorted[i + 1] == 0:
            continue
        if dev_sorted[i] / dev_sorted[i + 1] >= gap_ratio:
            elbow = i
            break

    hotspots = []
    if elbow is not None:
        for i in range(elbow + 1):
            r_int, c_int = np.unravel_index(flat_idx_sorted[i], interior.shape)
            r, c = r_int + border, c_int + border
            hotspots.append((r, c, float(gain[r, c]), float(dev_sorted[i])))

    return hotspots, median, dev_sorted, flat_idx_sorted, interior.shape


def save_diagnostic_plot(gain, hotspots, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    vlo, vhi = np.percentile(gain, [1, 99])
    ax.imshow(gain, cmap="gray", vmin=vlo, vmax=vhi)
    for r, c, val, dev in hotspots:
        ax.plot(c, r, "o", markerfacecolor="none", markeredgecolor="red", markersize=14, markeredgewidth=1.5)
    ax.set_title(f"{os.path.basename(out_path)}\n{len(hotspots)} hotspot(s) found")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output: parameter file + natural-language log
# ---------------------------------------------------------------------------
def shortest_reproducing_command(gain_path, values):
    parts = [os.path.basename(sys.argv[0]) if sys.argv else "ot_find_hotspots.py", gain_path]
    for name, typ, default, _grp, _help in OPTIONS:
        if name in ("input", "defaults"):
            continue
        val = values[name]
        if val == default:
            continue
        # shortest unique prefix for this option, computed against the full name list
        for n in range(1, len(name) + 1):
            prefix = name[:n]
            if sum(1 for other in _NAMES if other.startswith(prefix)) == 1:
                break
        flag = f"-{prefix}"
        if typ is bool:
            parts.append(flag)
        else:
            parts.append(f"{flag} {val}")
    return " ".join(parts)


def write_outputs(basename, gain_path, values, hotspots):
    param_path = f"{basename}_params.txt"
    with open(param_path, "w") as fh:
        fh.write(f"# Parameters actually used for this run of ot_find_hotspots.py on {gain_path}\n")
        for name, typ, default, _grp, _help in OPTIONS:
            if name in ("input", "defaults"):
                continue
            val = values[name]
            if typ is bool:
                fh.write(f"{name}={'true' if val else 'false'}\n")
            else:
                fh.write(f"{name}={'' if val is None else val}\n")
    print(f"Wrote parameter file: {param_path}")

    log_path = f"{basename}.log"
    repro_cmd = shortest_reproducing_command(gain_path, values)
    with open(log_path, "w") as fh:
        fh.write(f"Ran a hot-pixel scan on {gain_path}.\n")
        fh.write(f"A {values['border']}-pixel border was excluded from the scan, and pixels were "
                 f"flagged as hotspots when a consecutive-deviation gap of at least "
                 f"{values['gap_ratio']}x was found in the sorted deviations from the median.\n")
        fh.write(f"{len(hotspots)} hotspot(s) were found.\n")
        for r, c, val, dev in hotspots:
            fh.write(f"  - (row {r}, col {c}): value {val:.4f}, |deviation from median| {dev:.4f}\n")
        fh.write("\n")
        fh.write("Shortest command that reproduces this run:\n")
        fh.write(f"  {repro_cmd}\n")
    print(f"Wrote log file: {log_path}")

    # Derivative-input metadata file: per project convention, a script that
    # feeds a dependent second script (e.g. ot_patch_gain_reference.py) passes
    # its computed results forward automatically via a file, rather than
    # requiring the user to retype coordinates found here.
    meta_path = f"{basename}_hotspots.meta"
    with open(meta_path, "w") as fh:
        fh.write(f"# Hotspot metadata for {gain_path}, generated by ot_find_hotspots.py\n")
        fh.write("# Consumed automatically by ot_patch_gain_reference.py's -metadata option\n")
        fh.write(f"gain_path={gain_path}\n")
        fh.write(f"n_hotspots={len(hotspots)}\n")
        for i, (r, c, val, dev) in enumerate(hotspots, 1):
            fh.write(f"hotspot_{i}_row={r}\n")
            fh.write(f"hotspot_{i}_col={c}\n")
            fh.write(f"hotspot_{i}_value={val}\n")
    print(f"Wrote hotspot metadata file: {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    argv = sys.argv[1:]

    if len(argv) == 0:
        print_help()
        print()
        write_sample_param_file("ot_find_hotspots")
        sys.exit(0)

    parsed_gain_path, values, explicit = parse_args(argv)
    if parsed_gain_path == "HELP":
        print_help()
        sys.exit(0)

    gain_path, values = parsed_gain_path, values

    if gain_path is None:
        print("Error: no gain-reference file given.")
        print("Run with no arguments to see the full help menu.")
        sys.exit(2)

    # -input: fill in anything not explicitly set on the command line
    if values.get("input"):
        if "defaults" in explicit and values["defaults"]:
            print(f"Note: -defaults was also given; ignoring -input file '{values['input']}' "
                  f"for the border/gap_ratio/n_show settings.")
        else:
            loaded = load_param_file(values["input"])
            for k, v in loaded.items():
                if k not in explicit:
                    values[k] = v
            print(f"Loaded settings from parameter file: {values['input']}")

    # -defaults: force the detection parameters back to the built-in defaults
    if values["defaults"]:
        for name in ("border", "gap_ratio", "n_show"):
            values[name] = _DEFAULTS[name]
        print("Using built-in defaults for border / gap_ratio / n_show.")

    basename = values["output"] or os.path.splitext(os.path.basename(gain_path))[0]

    print(f"Loading gain reference: {gain_path}")
    with mrcfile.open(gain_path, permissive=True) as m:
        gain = m.data.astype(np.float64)
    print(f"  shape: {gain.shape}   min/max/median: {gain.min():.6g} / {gain.max():.6g} / {np.median(gain):.6g}")
    print(f"Excluding a {values['border']}-pixel border, then scanning the interior for hotspots "
          f"(no additional processes spawned; a single-threaded NumPy scan of an array this size "
          f"typically completes in well under a second).")

    hotspots, median, dev_sorted, flat_idx_sorted, interior_shape = find_hotspots(
        gain, values["border"], values["gap_ratio"]
    )

    print()
    print(f"Top {values['n_show']} interior pixels by |value - median| (median={median:.6g}):")
    for i in range(min(values["n_show"], len(dev_sorted))):
        r_int, c_int = np.unravel_index(flat_idx_sorted[i], interior_shape)
        r, c = r_int + values["border"], c_int + values["border"]
        print(f"  ({r}, {c})  value={gain[r, c]:.4f}  |dev|={dev_sorted[i]:.4f}")

    print()
    if hotspots:
        print(f"Hotspot(s) found via gap detection (ratio >= {values['gap_ratio']}):")
        for r, c, val, dev in hotspots:
            print(f"  ({r}, {c})  value={val:.4f}  |dev|={dev:.4f}")
    else:
        print(f"No clear elbow found at gap-ratio {values['gap_ratio']} -- deviations decay smoothly, "
              f"no standout hotspot in the interior region. Try lowering -gap_ratio if you expect a subtler defect.")

    if values["plot"]:
        if not _HAVE_MATPLOTLIB:
            print()
            print("Skipping -plot: matplotlib is not installed.")
            print("  Linux / macOS, user-only:   pip install --user matplotlib")
            print("  Linux / macOS, system-wide: sudo pip install matplotlib")
            print("  Externally-managed Python:  pip install --break-system-packages matplotlib")
        else:
            plot_path = f"{basename}_diagnostic.png"
            save_diagnostic_plot(gain, hotspots, plot_path)
            print(f"Wrote diagnostic plot: {plot_path}")

    write_outputs(basename, gain_path, values, hotspots)


if __name__ == "__main__":
    main()
