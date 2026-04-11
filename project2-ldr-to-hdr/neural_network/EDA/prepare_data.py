import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import get_exif

BRACKETED_DIR = ROOT / "data" / "images" / "Bracketed_images"
LDR_DIR       = ROOT / "data" / "images" / "LDR"
OUT_DIR       = ROOT / "neural_network" / "nn_data" / "selected"
OUT_LDR       = OUT_DIR / "LDR"
OUT_UNDER     = OUT_DIR / "Bracketed_images-27"
OUT_OVER      = OUT_DIR / "Bracketed_images+27"

BOLD   = "\033[1m";  RESET  = "\033[0m"
GREEN  = "\033[92m"; YELLOW = "\033[93m"
RED    = "\033[91m"

TARGET_UNDER = -2.7
TARGET_OVER  = +2.7
EV_TOLERANCE = 0.05


def ev_from_exif(exif: dict) -> float | None:
    val = exif.get("ExposureBiasValue")
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def find_bracketed_targets(scene_dir: Path) -> dict:
    """Scan scene folder, return paths of images matching EV -2.7 and/or +2.7."""
    found = {}
    for img_path in sorted(scene_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        try:
            exif = get_exif(str(img_path))
        except ValueError:
            continue
        ev = ev_from_exif(exif)
        if ev is None:
            continue
        if abs(ev - TARGET_UNDER) <= EV_TOLERANCE and "under" not in found:
            found["under"] = img_path
        elif abs(ev - TARGET_OVER) <= EV_TOLERANCE and "over" not in found:
            found["over"] = img_path
    return found


def find_ldr(scene: str) -> Path | None:
    for f in LDR_DIR.iterdir():
        if f.name.upper().startswith(scene.upper()) and \
           f.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}:
            return f
    return None


def run(dry_run: bool = False):
    OUT_LDR.mkdir(parents=True, exist_ok=True)
    OUT_UNDER.mkdir(parents=True, exist_ok=True)
    OUT_OVER.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Data preparation — scanning and copying{RESET}")
    if dry_run:
        print(f"{YELLOW}  DRY RUN — no files will be copied{RESET}")

    print(f"{BOLD}{'='*60}{RESET}\n")

    stats = {"ldr": 0, "under": 0, "over": 0, "no_targets": []}

    scene_dirs = sorted([d for d in BRACKETED_DIR.iterdir() if d.is_dir()])

    for scene_dir in scene_dirs:
        scene   = scene_dir.name
        targets = find_bracketed_targets(scene_dir)
        ldr     = find_ldr(scene)

        if not targets:
            stats["no_targets"].append(scene)
            print(f"  {YELLOW}⚠ {scene}: no EV=±2.7 images found, skipping{RESET}")
            continue

        parts = []

        if "under" in targets:
            src = targets["under"]
            dst = OUT_UNDER / f"{scene}-27{src.suffix}"
            if not dry_run:
                shutil.copy2(src, dst)
            parts.append(f"{GREEN}under ✓{RESET}")
            stats["under"] += 1
        else:
            parts.append(f"{YELLOW}under –{RESET}")

        if "over" in targets:
            src = targets["over"]
            dst = OUT_OVER / f"{scene}+27{src.suffix}"
            if not dry_run:
                shutil.copy2(src, dst)
            parts.append(f"{GREEN}over ✓{RESET}")
            stats["over"] += 1
        else:
            parts.append(f"{YELLOW}over –{RESET}")

        if ldr:
            dst = OUT_LDR / ldr.name
            if not dry_run:
                shutil.copy2(ldr, dst)
            parts.append(f"{GREEN}LDR ✓{RESET}")
            stats["ldr"] += 1
        else:
            parts.append(f"{RED}LDR ✗{RESET}")

        print(f"  {BOLD}{scene}{RESET}  {' | '.join(parts)}")

    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  Scenes processed       : {len(scene_dirs) - len(stats['no_targets'])}")
    print(f"  Under (-2.7) copied    : {stats['under']}")
    print(f"  Over  (+2.7) copied    : {stats['over']}")
    print(f"  LDR copied             : {stats['ldr']}")
    if stats["no_targets"]:
        print(f"  {YELLOW}Skipped (no EV match)  : {', '.join(stats['no_targets'])}{RESET}")
    print(f"\n{BOLD}{'='*60}{RESET}\n")