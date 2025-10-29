# save as download_aro_official.py and run:  python download_aro_official.py
import os, json, zipfile, sys
from pathlib import Path

print("="*60)
print("ARO DATASET DOWNLOADER (tool-free, cross-platform)")
print("="*60)

# --- Constants from the official ARO code ---
IMAGES_ID = "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9"           # vgr_vga_images.zip
REL_JSON_ID = "1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3"         # visual_genome_relation.json
ATT_JSON_ID = "13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw"         # visual_genome_attribution.json

root = Path("data/aro")
root.mkdir(parents=True, exist_ok=True)
images_zip = root/"vgr_vga_images.zip"
rel_json   = root/"visual_genome_relation.json"
att_json   = root/"visual_genome_attribution.json"

def need(p): return not p.exists()

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    os.system(f"{sys.executable} -m pip install gdown --quiet")
    import gdown

def dl(id_, out):
    if out.exists():
        print(f"✓ Found {out.name}")
        return
    print(f"↓ Downloading {out.name} ...")
    gdown.download(id=id_, output=str(out), quiet=False, use_cookies=False)

# 1) Download files
dl(IMAGES_ID, images_zip)
dl(REL_JSON_ID, rel_json)
dl(ATT_JSON_ID, att_json)

# 2) Extract images zip (creates root/images/…)
images_dir = root/"images"
if not images_dir.exists():
    print("↪ Extracting images…")
    with zipfile.ZipFile(images_zip, "r") as zf:
        zf.extractall(root)
else:
    print("✓ Images already extracted")

# 3) Sanity checks
def count_images():
    n = 0
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            n += 1
    return n

ok = images_dir.exists() and images_dir.is_dir()
nimg = count_images() if ok else 0
print(f"✓ Done. images dir: {images_dir}  (files: {nimg})")
print(f"   JSONs: {rel_json.exists()} (relation), {att_json.exists()} (attribution)")

if not ok or nimg == 0:
    raise SystemExit("❌ Images missing after extraction; see notes below.")
print("✅ ARO VG-Relation & VG-Attribution are ready under data/aro/")
