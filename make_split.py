"""P0: 生成训练/验证数据划分清单，验证图像可解码，输出 split_manifest.json。"""
import json, os, hashlib
import numpy as np
import cv2

SEED = 1234
DIV2K_DIR = "train/DIV2K_train_HR/DIV2K_train_HR"
URBAN_DIR  = "train/Urban100/Urban100-master/HR"
OUT_FILE   = "split_manifest.json"

rng = np.random.default_rng(SEED)

def list_images(d):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts)
    return [os.path.join(d, f) for f in files]

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()

def verify_and_info(paths, label):
    bad, info = [], []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            bad.append(p); continue
        h, w = img.shape[:2]
        info.append({"path": p, "h": h, "w": w, "hash": file_hash(p)})
    if bad:
        print(f"  [WARN] {label}: {len(bad)} files failed to decode: {bad[:3]}")
    return info

# --- load ---
div2k  = list_images(DIV2K_DIR)
urban  = list_images(URBAN_DIR)
print(f"DIV2K: {len(div2k)}, Urban100: {len(urban)}")

# --- shuffle independently ---
div2k_idx  = rng.permutation(len(div2k)).tolist()
urban_idx  = rng.permutation(len(urban)).tolist()

div2k_train = [div2k[i] for i in div2k_idx[:720]]
div2k_val   = [div2k[i] for i in div2k_idx[720:]]
urban_train = [urban[i] for i in urban_idx[:90]]
urban_val   = [urban[i] for i in urban_idx[90:]]

print(f"Split: DIV2K {len(div2k_train)}/{len(div2k_val)}, "
      f"Urban {len(urban_train)}/{len(urban_val)}")

# --- verify ---
print("Verifying images (this may take a moment)...")
all_info = {}
for label, paths in [("div2k_train", div2k_train), ("div2k_val", div2k_val),
                      ("urban_train", urban_train), ("urban_val", urban_val)]:
    print(f"  {label} ...")
    all_info[label] = verify_and_info(paths, label)

# --- duplicate check across all splits ---
all_hashes = []
for k, v in all_info.items():
    for item in v:
        all_hashes.append((item["hash"], k, item["path"]))
hash_map = {}
dups = []
for h, split, p in all_hashes:
    if h in hash_map:
        dups.append((p, split, hash_map[h]))
    else:
        hash_map[h] = (split, p)
if dups:
    print(f"[WARN] {len(dups)} duplicate files found across splits:")
    for d in dups[:5]:
        print(f"  {d}")
else:
    print("No duplicates found across splits.")

# --- size summary ---
for k, v in all_info.items():
    hs = [x["h"] for x in v]
    ws = [x["w"] for x in v]
    print(f"  {k}: n={len(v)}, h=[{min(hs)},{max(hs)}], w=[{min(ws)},{max(ws)}]")

# --- save manifest ---
manifest = {
    "seed": SEED,
    "split": {k: [x["path"] for x in v] for k, v in all_info.items()},
    "counts": {k: len(v) for k, v in all_info.items()},
    "image_info": all_info,
}
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"\nManifest saved to {OUT_FILE}")
