import argparse
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--output", default="outputs/rule_stability.json")
    args = p.parse_args()
    models = []
    for path in args.models:
        with Path(path).open("rb") as handle:
            record = pickle.load(handle)
        if "model" not in record:
            raise ValueError("Rule stability requires fuzzy models")
        models.append(record)
    distances = []
    scores = []
    for first, second in combinations(models, 2):
        a = first["model"].centers.detach().numpy() * first["scale"] + first["center"]
        b = second["model"].centers.detach().numpy() * second["scale"] + second["center"]
        distance = cdist(a / first["scale"], b / first["scale"]) / np.sqrt(a.shape[1])
        i, j = linear_sum_assignment(distance)
        distances.append(float(distance[i, j].mean()))
        scores.append(float(np.exp(-distance[i, j]).sum() / max(len(a), len(b))))
    output = dict(pairs=len(scores), matched_rule_stability=float(np.mean(scores)) if scores else None, matched_center_distance=float(np.mean(distances)) if distances else None)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
