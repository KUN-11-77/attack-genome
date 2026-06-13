import sys
p = "/data3/khsong/cogatedrive/outputs/ag_tier_cpp_shard0/summary.txt"
with open(p) as f:
    text = f.read()
print("file size:", len(text), "chars, lines:", len(text.splitlines()))
hits = 0
for i, line in enumerate(text.splitlines()):
    line2 = line.strip()
    if not line2 or line2.startswith("#"):
        continue
    parts = [x.strip() for x in line2.split("|")]
    if len(parts) >= 3 and parts[0] in ["CNN", "DINO", "TF"]:
        hits += 1
        if hits <= 3:
            print("  HIT:", repr(parts[0]), repr(parts[1][:30]))
print("TOTAL hits:", hits)
