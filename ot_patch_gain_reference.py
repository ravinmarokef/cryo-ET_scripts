#!/usr/bin/env python3
# ot_patch_gain_reference.py -- interpolates over known hot-pixel defects in a gain reference
# Created 20260901 (David Jurkowitz - Gan Lab using Claude)
# Revised 20260901 18:50
#
# -----------------------------------------------------------------------------
# Generated with the assistance of Claude (Anthropic).
# Model: Claude Sonnet 5 (claude-sonnet-5)        Date: 2026-09-01
# Reviewed, tested, and validated by the Gan Lab (Anaphase); the authors take
# full responsibility for the correctness of this code.
# Lab: https://github.com/anaphaze | https://www.anaphase.org
# -----------------------------------------------------------------------------
"""
ot_patch_gain_reference.py

Interpolates over known hot-pixel defects in a gain reference (.mrc), using
the local-neighborhood median at each defect coordinate. After patching, the
result is re-scanned with the same gap-based method as ot_find_hotspots.py
to confirm no other similarly extreme pixel was missed.

TWO-SCRIPT PIPELINE
  Coordinates are read automatically from the <basename>_hotspots.meta file
  written by ot_find_hotspots.py -- you should not need to retype them.
  If no metadata file is found (and -coords isn't given), this script tells
  you to run ot_find_hotspots.py first, or supply coordinates manually.

  Coordinate resolution order:
    1. -coords, if given (manual override)
    2. -metadata <path>, if given (explicit metadata file)
    3. auto-derived <gain_basename>_hotspots.meta, checked next to the gain
       file and in the current directory

DEPENDENCIES
  numpy, mrcfile (required); matplotlib (optional, only for -plot).
  If missing, running this script will print install instructions for
  Linux and macOS, both user-only and system-wide.

SCRIPT IS NON-INTERACTIVE. Run with no arguments to see the full help
menu and a sample input-parameter file.

NOTE: this copy was written but deliberately not executed against real
data -- please verify the patched output yourself (e.g. -plot, or -simulate
for a dry run) before trusting it on anything you care about.
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
# calibration file, not a biophysical/sample model.
#
# NOTE: canonical names below are chosen so every first letter is unique,
# which is what makes single-character abbreviations (-m, -c, -r, -b, -g,
# -o, -i, -d, -s, -p) unambiguous. If you add new options, keep that
# property or the parser will (correctly) refuse to guess.
# ---------------------------------------------------------------------------
OPTIONS = [
    ("metadata",  str,   None, "ALGORITHM", "Path to a hotspot metadata file from ot_find_hotspots.py. Default: auto-derived next to the gain file as <gain_basename>_hotspots.meta."),
    ("coords",    str,   None, "ALGORITHM", "Manual override: semicolon-separated row,col pairs to patch, e.g. '932,1602;100,200'. Takes precedence over -metadata."),
    ("radius",    int,   3,    "ALGORITHM", "Neighborhood half-width used for median interpolation around each patched pixel."),
    ("border",    int,   5,    "ALGORITHM", "Border pixels excluded when re-scanning the patched array for any remaining hotspots."),
    ("gap_ratio", float, 3.0,  "ALGORITHM", "Gap-ratio threshold for the post-patch re-scan (same method as ot_find_hotspots.py)."),
    ("output",    str,   None, "ALGORITHM", "Basename override for all output files (patched .mrc, parameter file, log, diagnostic plot). Default: '<gain_basename>_fixed'."),
    ("input",     str,   None, "ALGORITHM", "Path to an input-parameter file (key=value per line) to load settings from."),
    ("defaults",  bool,  False,"ALGORITHM", "Ignore any parameter file / other flags above and use the built-in defaults for radius/border/gap_ratio."),
    ("simulate",  bool,  False,"ALGORITHM", "Dry run: report what would be patched and re-scanned without writing the patched .mrc file."),
    ("plot",      bool,  False,"GRAPHICAL", "Save a diagnostic PNG marking patched location(s) (green) and any remaining hotspot(s) found on re-scan (red)."),
]
_NAMES = [o[0] for o in OPTIONS]
_TYPES = {o[0]: o[1] for o in OPTIONS}
_DEFAULTS = {o[0]: o[2] for o in OPTIONS}


# ---------------------------------------------------------------------------
# Minimal-unique-prefix single-dash option parsing (same convention as
# ot_find_hotspots.py)
# ---------------------------------------------------------------------------
def resolve_option(token, names):
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

        if gain_path is None:
            gain_path = tok
            i += 1
        else:
            print(f"Error: unexpected extra argument '{tok}'.")
            sys.exit(2)

    return gain_path, values, explicit


def load_param_file(path):
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
    print("  ot_patch_gain_reference.py GAIN_FILE.mrc [options]")
    print()
    for group in ("BIOPHYSICAL", "ALGORITHM/PARALLELIZATION", "GRAPHICAL"):
        print(group)
        members = [o for o in OPTIONS if o[3] == group or group.startswith(o[3])]
        if not members:
            print("  (none -- this script has no biophysical/sample-model parameters)")
        for name, typ, default, _grp, help_text in members:
            flag = f"-{name}"
            defstr = "off" if typ is bool else str(default)
            print(f"  {flag:<12} (default: {defstr})  {help_text}")
        if group == "ALGORITHM/PARALLELIZATION":
            print("  Parallelization note: patching a handful of known coordinates and")
            print("  re-scanning one array is already fast single-threaded. If you need")
            print("  to patch many gain-reference files, parallelize ACROSS FILES, e.g.:")
            print("    ls *.mrc | xargs -P4 -I{} python3 ot_patch_gain_reference.py {} -d")
        print()
    print("  -h / --help    Show this help menu.")
    print()
    print("INPUT-PARAMETER FILE FORMAT")
    print("  Plain text, one 'key=value' pair per line, '#' for comments.")
    print("  Keys accept the same minimal-unique abbreviations as the command")
    print("  line. Load one with -input <file>.")
    print()
    print("EXAMPLE (shortest command that reproduces the all-defaults run,")
    print("assuming a <gain_basename>_hotspots.meta file already exists):")
    print("  ot_patch_gain_reference.py GainReference.mrc -d")


def write_sample_param_file(basename):
    path = f"{basename}_defaults.param"
    with open(path, "w") as fh:
        fh.write("# Sample input-parameter file for ot_patch_gain_reference.py, auto-generated.\n")
        fh.write("# Edit values as needed, then run:\n")
        fh.write(f"#   ot_patch_gain_reference.py GainReference.mrc -input {path}\n")
        for name, typ, default, group, _help in OPTIONS:
            if name in ("input", "defaults"):
                continue
            if typ is bool:
                fh.write(f"{name}={'true' if default else 'false'}\n")
            else:
                fh.write(f"{name}={'' if default is None else default}\n")
    print(f"Wrote sample parameter file: {path}")
    return path


# ---------------------------------------------------------------------------
# Coordinate resolution
# ---------------------------------------------------------------------------
def parse_coords_string(s):
    coords = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            r_str, c_str = chunk.split(",")
            coords.append((int(r_str.strip()), int(c_str.strip())))
        except ValueError:
            print(f"Error: could not parse coordinate pair '{chunk}' in -coords "
                  f"(expected 'row,col').")
            sys.exit(2)
    return coords


def load_hotspot_metadata(path):
    kv = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = (s.strip() for s in line.split("=", 1))
            kv[key] = val

    n = int(kv.get("n_hotspots", 0))
    coords = []
    for i in range(1, n + 1):
        try:
            r = int(kv[f"hotspot_{i}_row"])
            c = int(kv[f"hotspot_{i}_col"])
        except KeyError:
            print(f"Warning: metadata file {path} claims n_hotspots={n} but is "
                  f"missing entry {i}; stopping there.")
            break
        coords.append((r, c))
    return coords


def resolve_coordinates(gain_path, values):
    if values["coords"]:
        coords = parse_coords_string(values["coords"])
        print(f"Using {len(coords)} manually-specified coordinate(s) from -coords.")
        return coords

    meta_path = values["metadata"]
    if not meta_path:
        gain_dir = os.path.dirname(gain_path) or "."
        auto_name = os.path.splitext(os.path.basename(gain_path))[0] + "_hotspots.meta"
        candidates = [os.path.join(gain_dir, auto_name), auto_name]
        for c in candidates:
            if os.path.isfile(c):
                meta_path = c
                break
        if not meta_path:
            print("Error: no hotspot coordinates given. No -coords supplied, and no "
                  f"metadata file found at either of:")
            for c in candidates:
                print(f"    {c}")
            print("Run ot_find_hotspots.py on this gain file first (it writes this "
                  "file automatically), or supply -coords 'row,col;...' manually.")
            sys.exit(2)

    if not os.path.isfile(meta_path):
        print(f"Error: metadata file not found: {meta_path}")
        sys.exit(2)

    coords = load_hotspot_metadata(meta_path)
    print(f"Loaded {len(coords)} hotspot coordinate(s) from metadata file: {meta_path}")
    if len(coords) == 0:
        print("Note: metadata file reports zero hotspots -- nothing to patch. "
              "Exiting without writing an output file.")
        sys.exit(0)
    return coords


# ---------------------------------------------------------------------------
# Patch + re-scan logic
# ---------------------------------------------------------------------------
def patch_pixels(original, coords, radius):
    """Interpolate each listed coordinate from the median of its unpatched,
    non-defective neighborhood. Neighborhoods are always drawn from the
    original (pre-patch) array, and any neighbor pixel that is itself a
    listed defect is excluded, so patching one hotspot never contaminates
    -- or is contaminated by -- another nearby one."""
    patched = original.copy()
    coord_set = set(coords)
    h, w = original.shape

    for (r, c) in coords:
        r0, r1 = max(0, r - radius), min(h, r + radius + 1)
        c0, c1 = max(0, c - radius), min(w, c + radius + 1)
        neighborhood = []
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                if (rr, cc) == (r, c):
                    continue
                if (rr, cc) in coord_set:
                    continue  # don't let another known defect skew the median
                neighborhood.append(original[rr, cc])
        if not neighborhood:
            print(f"Error: pixel ({r}, {c}) has no valid (non-defect) neighbors "
                  f"within radius {radius}; increase -radius.")
            sys.exit(1)
        patched[r, c] = np.median(neighborhood)

    return patched


def find_hotspots(gain, border, gap_ratio):
    """Same gap-based detection method as ot_find_hotspots.py, used here to
    verify the patch worked and nothing else was missed."""
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

    return hotspots


def save_diagnostic_plot(patched, patched_coords, remaining_hotspots, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    vlo, vhi = np.percentile(patched, [1, 99])
    ax.imshow(patched, cmap="gray", vmin=vlo, vmax=vhi)
    for r, c in patched_coords:
        ax.plot(c, r, "s", markerfacecolor="none", markeredgecolor="lime", markersize=14, markeredgewidth=1.5)
    for r, c, val, dev in remaining_hotspots:
        ax.plot(c, r, "o", markerfacecolor="none", markeredgecolor="red", markersize=16, markeredgewidth=1.5)
    ax.set_title(f"{os.path.basename(out_path)}\n"
                 f"{len(patched_coords)} patched (green), {len(remaining_hotspots)} remaining (red)")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output: parameter file + natural-language log + chained metadata
# ---------------------------------------------------------------------------
def shortest_reproducing_command(gain_path, values):
    parts = [os.path.basename(sys.argv[0]) if sys.argv else "ot_patch_gain_reference.py", gain_path]
    for name, typ, default, _grp, _help in OPTIONS:
        if name in ("input", "defaults"):
            continue
        val = values[name]
        if val == default:
            continue
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


def write_outputs(basename, gain_path, values, patched_coords, remaining_hotspots, wrote_mrc):
    param_path = f"{basename}_params.txt"
    with open(param_path, "w") as fh:
        fh.write(f"# Parameters actually used for this run of ot_patch_gain_reference.py on {gain_path}\n")
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
        if wrote_mrc:
            fh.write(f"Patched {len(patched_coords)} hot pixel(s) in {gain_path} using local-neighborhood "
                     f"median interpolation (radius {values['radius']}), and wrote the result to "
                     f"{basename}.mrc.\n")
        else:
            fh.write(f"Simulated (dry run, -simulate) patching {len(patched_coords)} hot pixel(s) in "
                     f"{gain_path} using local-neighborhood median interpolation (radius "
                     f"{values['radius']}). No output .mrc file was written.\n")
        for r, c in patched_coords:
            fh.write(f"  - patched (row {r}, col {c})\n")
        fh.write(f"\nRe-scanned the patched array ({values['border']}-pixel border excluded, "
                 f"gap-ratio {values['gap_ratio']}) and found {len(remaining_hotspots)} remaining hotspot(s).\n")
        for r, c, val, dev in remaining_hotspots:
            fh.write(f"  - (row {r}, col {c}): value {val:.4f}, |deviation from median| {dev:.4f}\n")
        fh.write("\n")
        fh.write("Shortest command that reproduces this run:\n")
        fh.write(f"  {repro_cmd}\n")
    print(f"Wrote log file: {log_path}")

    # Chained metadata: if the re-scan still found something, write it out in
    # the same format ot_find_hotspots.py uses, so it can be fed straight
    # back into another round of this same script if needed.
    meta_path = f"{basename}_hotspots.meta"
    meta_gain_path = f"{basename}.mrc" if wrote_mrc else gain_path
    with open(meta_path, "w") as fh:
        if wrote_mrc:
            fh.write(f"# Post-patch hotspot metadata for {meta_gain_path}, generated by ot_patch_gain_reference.py\n")
        else:
            fh.write(f"# Post-patch hotspot metadata from a -simulate (dry run) on {meta_gain_path}; "
                     f"no patched .mrc file exists yet, this reflects the original file.\n")
        fh.write(f"gain_path={meta_gain_path}\n")
        fh.write(f"n_hotspots={len(remaining_hotspots)}\n")
        for i, (r, c, val, dev) in enumerate(remaining_hotspots, 1):
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
        write_sample_param_file("ot_patch_gain_reference")
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

    if values.get("input"):
        if values["defaults"]:
            print(f"Note: -defaults was also given; ignoring -input file '{values['input']}' "
                  f"for the radius/border/gap_ratio settings.")
        else:
            loaded = load_param_file(values["input"])
            for k, v in loaded.items():
                if k not in explicit:
                    values[k] = v
            print(f"Loaded settings from parameter file: {values['input']}")

    if values["defaults"]:
        for name in ("radius", "border", "gap_ratio"):
            values[name] = _DEFAULTS[name]
        print("Using built-in defaults for radius / border / gap_ratio.")

    coords = resolve_coordinates(gain_path, values)

    basename = values["output"] or f"{os.path.splitext(os.path.basename(gain_path))[0]}_fixed"

    print(f"Loading gain reference: {gain_path}")
    with mrcfile.open(gain_path, permissive=True) as m:
        gain = m.data.astype(np.float64)
    print(f"  shape: {gain.shape}   min/max/median: {gain.min():.6g} / {gain.max():.6g} / {np.median(gain):.6g}")

    print(f"Patching {len(coords)} pixel(s) with radius {values['radius']} "
          f"(no additional processes spawned; this completes essentially instantly "
          f"for a handful of coordinates).")
    patched = patch_pixels(gain, coords, values["radius"])

    print(f"Re-scanning the patched array (excluding a {values['border']}-pixel border, "
          f"gap-ratio {values['gap_ratio']}) to confirm nothing else was missed...")
    remaining_hotspots = find_hotspots(patched, values["border"], values["gap_ratio"])
    if remaining_hotspots:
        print(f"  {len(remaining_hotspots)} hotspot(s) still present after patching:")
        for r, c, val, dev in remaining_hotspots:
            print(f"    ({r}, {c})  value={val:.4f}  |dev|={dev:.4f}")
    else:
        print("  No remaining hotspots found -- patch looks clean.")

    wrote_mrc = False
    if values["simulate"]:
        print()
        print(f"-simulate given: NOT writing {basename}.mrc. Rerun without -simulate to write it.")
    else:
        out_mrc_path = f"{basename}.mrc"
        with mrcfile.new(out_mrc_path, overwrite=True) as m:
            m.set_data(patched.astype(np.float32))
        print(f"Wrote patched gain reference: {out_mrc_path}")
        wrote_mrc = True

    if values["plot"]:
        if not _HAVE_MATPLOTLIB:
            print()
            print("Skipping -plot: matplotlib is not installed.")
            print("  Linux / macOS, user-only:   pip install --user matplotlib")
            print("  Linux / macOS, system-wide: sudo pip install matplotlib")
            print("  Externally-managed Python:  pip install --break-system-packages matplotlib")
        else:
            plot_path = f"{basename}_diagnostic.png"
            save_diagnostic_plot(patched, coords, remaining_hotspots, plot_path)
            print(f"Wrote diagnostic plot: {plot_path}")

    write_outputs(basename, gain_path, values, coords, remaining_hotspots, wrote_mrc)


if __name__ == "__main__":
    main()
