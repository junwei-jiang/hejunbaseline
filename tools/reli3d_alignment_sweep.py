#!/usr/bin/env python3
import argparse
import itertools
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_cmd', type=str, required=True,
                   help='Base command excluding tested flags, e.g. "python main.py ..."')
    p.add_argument('--run', action='store_true', help='Actually execute commands')
    p.add_argument('--limit', type=int, default=8, help='Max combinations to run/print')
    args = p.parse_args()

    combos = list(itertools.product([0, 1], [0, 1], ['dataset', 'center'], ['xy', 'scalar_x']))
    combos = combos[: max(1, int(args.limit))]

    for idx, (no_conv, repaired, principal_mode, fov_mode) in enumerate(combos):
        cmd = (
            f"{args.base_cmd} "
            f"--reli3d_no_source_view_conversion " if no_conv else f"{args.base_cmd} "
        )
        cmd += (
            f"--reli3d_mapper_dataset_repaired {repaired} "
            f"--reli3d_export_principal_mode {principal_mode} "
            f"--reli3d_export_fov_mode {fov_mode} "
            f"--reli3d_dump_camera_debug "
            f"--output_dir ./output_sweep/c{idx:02d}_conv{1-no_conv}_rep{repaired}_{principal_mode}_{fov_mode}"
        )

        print(f"\n[combo {idx}] {cmd}")
        if args.run:
            proc = subprocess.run(cmd, shell=True)
            if proc.returncode != 0:
                print(f"[combo {idx}] failed with code {proc.returncode}")
                break


if __name__ == '__main__':
    main()
