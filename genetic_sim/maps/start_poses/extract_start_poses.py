"""
Build start_poses.csv from the racetracks' centerline files.

For each track folder in maps/f1tenth_racetracks, reads the first two points of
<Name>_centerline.csv and writes one row:

    track, map, x, y, theta

- track : folder name (what you pass to nn_controller.py --map)
- map   : map path without extension, relative to maps/f1tenth_racetracks
- x, y  : first centerline point (m)
- theta : heading from the first point towards the second (rad)

Once start_poses.csv exists, the centerline/raceline/DonkeySim files are no
longer needed by the controller.

    python extract_start_poses.py
"""

import csv
import glob
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(HERE, '..', 'f1tenth_racetracks')
OUT_PATH = os.path.join(HERE, 'start_poses.csv')


def first_two_points(path):
    points = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            x, y = (float(v) for v in line.split(',')[:2])
            points.append((x, y))
            if len(points) == 2:
                return points
    raise ValueError(f'{path}: fewer than two points')


def main():
    rows = []
    for track_dir in sorted(os.listdir(MAPS_DIR)):
        full_dir = os.path.join(MAPS_DIR, track_dir)
        centerlines = glob.glob(os.path.join(full_dir, '*_centerline.csv'))
        maps = glob.glob(os.path.join(full_dir, '*_map.yaml'))
        if not centerlines or not maps:
            continue

        (x0, y0), (x1, y1) = first_two_points(centerlines[0])
        theta = math.atan2(y1 - y0, x1 - x0)
        map_rel = os.path.relpath(maps[0], MAPS_DIR)[:-len('.yaml')].replace(os.sep, '/')
        rows.append((track_dir, map_rel, x0, y0, round(theta, 6)))

    # Centerline files may have been deleted after extraction; keep the existing CSV.
    if not rows:
        print(f'No *_centerline.csv files found under {MAPS_DIR}; {OUT_PATH} left unchanged')
        return

    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['track', 'map', 'x', 'y', 'theta'])
        writer.writerows(rows)
    print(f'Wrote {len(rows)} tracks to {OUT_PATH}')


if __name__ == '__main__':
    main()
