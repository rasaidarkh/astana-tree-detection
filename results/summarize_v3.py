"""Print final leaderboard of all 10 v3 experiments."""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

data = json.loads(open("results/v3_experiments.json").read())

def archive_short(p: str) -> str:
    if not p:
        return ""
    return p.replace("weights\\v3_runs\\", "").replace("weights/v3_runs/", "")

print()
print("=" * 130)
print(f'{"id":<32} {"time":>5} {"v2-Box":>7} {"v3-Box":>7} {"mrg-Box":>8} {"mrg-Mask":>9}  {"weights archive":<60}')
print("=" * 130)

sort = sorted(data, key=lambda r: r.get("metrics", {}).get("merged", {}).get("box_map50", -1), reverse=True)
for r in sort:
    if r.get("status") != "completed":
        print(f'{r["id"]:<32} FAIL: {r.get("error","?")[:60]}')
        continue
    m = r["metrics"]
    arch = archive_short(r.get("best_pt_archive", ""))
    print(f'{r["id"]:<32} {r["wall_time_min"]:>5.0f}m {m["v2-val"]["box_map50"]:>7.3f} {m["v3-val"]["box_map50"]:>7.3f} {m["merged"]["box_map50"]:>8.3f} {m["merged"]["mask_map50"]:>9.3f}  {arch:<60}')

print("=" * 130)
print()
print("Baselines (same merged val):")
print(f'  v2-finetune (pre-v3, old prod):  Box mAP50 = 0.167  Mask = 0.169')
print(f'  v3 run1 (yolov8x, old prod):     Box mAP50 = 0.268  Mask = 0.244')
print()
print("=== Top 3 by merged Box mAP50 ===")
for i, r in enumerate(sort[:3]):
    m = r["metrics"]
    print(f"#{i+1}: {r['id']}")
    print(f"    {r['description']}")
    print(f"    v2-val={m['v2-val']['box_map50']:.4f}  v3-val={m['v3-val']['box_map50']:.4f}  merged={m['merged']['box_map50']:.4f}  Mask={m['merged']['mask_map50']:.4f}")
    print(f"    archive: {r.get('best_pt_archive','?')}")
