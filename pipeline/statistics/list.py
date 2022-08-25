

import glob
import json
import statistics

stats = {}
for f in glob.glob("./*.json"):
    with open(f) as fr: 
        s = json.load(fr)
        for key, list in s.items():
            if not key in stats:
                stats[key] = []
            stats[key] += list

for key, list in stats.items():
    if key != "fps":
        list = [l * 1000 for l in list]
    print(f"{key} & {len(list):.2f} & {statistics.mean(list):.2f} & {statistics.median(list):.2f} & {statistics.stdev(list):.2f} \\\\")
        