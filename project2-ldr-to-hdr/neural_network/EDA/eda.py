import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import get_exif

BRACKETED_DIR = ROOT / "data" / "images" / "Bracketed_images"
LDR_DIR       = ROOT / "data" / "images" / "LDR"
TEST_SCENES   = {f"C{i:02d}" for i in range(40, 47)}

BOLD   = "\033[1m";  RESET  = "\033[0m"
RED    = "\033[91m"; YELLOW = "\033[93m"
GREEN  = "\033[92m"; CYAN   = "\033[96m"


def ev_from_exif(exif: dict):
    val = exif.get("ExposureBiasValue")
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def et_as_float(et) -> float | None:
    """Convert ExposureTime (IFDRational or tuple) to float seconds."""
    if et is None:
        return None
    try:
        return float(et)
    except Exception:
        return None


def scan_bracketed(bracketed_dir: Path = BRACKETED_DIR) -> dict:
    scenes = {}
    if not bracketed_dir.exists():
        print(f"[ERROR] Not found: {bracketed_dir}")
        return scenes

    for scene_dir in sorted(bracketed_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        images = sorted([
            f for f in scene_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        ])
        file_info = []
        for img_path in images:
            try:
                exif = get_exif(str(img_path))
            except ValueError:
                exif = {}
            file_info.append({
                "file":          img_path.name,
                "ev":            ev_from_exif(exif),
                "exposure_time": et_as_float(exif.get("ExposureTime")),
                "fnumber":       str(exif.get("FNumber")) if exif.get("FNumber") else None,
            })

        ev_values = sorted({round(f["ev"], 2) for f in file_info if f["ev"] is not None})
        scenes[scene_dir.name] = {
            "n_files":    len(images),
            "ev_values":  ev_values,
            "ev_missing": sum(1 for f in file_info if f["ev"] is None),
            "files":      file_info,
        }
    return scenes


def scan_ldr(ldr_dir: Path = LDR_DIR) -> dict:
    ldr_scenes = {}
    if not ldr_dir.exists():
        print(f"[ERROR] Not found: {ldr_dir}")
        return ldr_scenes
    for f in sorted(ldr_dir.iterdir()):
        if f.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}:
            scene = f.name.split("_")[0]
            ldr_scenes[scene] = f.name
    return ldr_scenes


def analyze_missing_ev_scenes(scenes: dict) -> dict:
    """
    For scenes where all EV metadata is missing but n_files == 9,
    sort files by ExposureTime and compute EV steps between consecutive shots.
    Returns a dict: scene -> analysis result.
    """
    results = {}
    for scene, info in scenes.items():
        if info["ev_missing"] != info["n_files"] or info["n_files"] != 9:
            continue

        files = info["files"]
        # filter files that have ExposureTime
        with_et = [f for f in files if f["exposure_time"] is not None]

        if len(with_et) != 9:
            results[scene] = {"status": "no_et", "files_with_et": len(with_et)}
            continue

        sorted_files = sorted(with_et, key=lambda f: f["exposure_time"])
        ets = [f["exposure_time"] for f in sorted_files]

        # EV diff between consecutive shots: log2(ET_next / ET_prev)
        diffs = [math.log2(ets[i + 1] / ets[i]) for i in range(len(ets) - 1)]
        avg_step  = sum(diffs) / len(diffs)
        min_step  = min(diffs)
        max_step  = max(diffs)
        consistent = all(abs(d - avg_step) < 0.15 for d in diffs)

        # EV of first shot relative to middle (shot index 4 = EV 0)
        ev_first = sum(diffs[:4]) * -1   # going backwards from middle
        ev_last  = sum(diffs[4:])        # going forward from middle

        results[scene] = {
            "status":      "ok" if consistent else "inconsistent",
            "sorted_files": [f["file"] for f in sorted_files],
            "ets":          ets,
            "diffs":        [round(d, 3) for d in diffs],
            "avg_step":     round(avg_step, 3),
            "min_step":     round(min_step, 3),
            "max_step":     round(max_step, 3),
            "consistent":   consistent,
            "ev_first_est": round(ev_first, 2),
            "ev_last_est":  round(ev_last,  2),
            "underexposed": sorted_files[0]["file"],   # shortest ET
            "normal":       sorted_files[4]["file"],   # middle
            "overexposed":  sorted_files[8]["file"],   # longest ET
        }
    return results


def print_report(scenes: dict, ldr_scenes: dict):
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  HDR-Eye EDA Report{RESET}")
    print(f"{BOLD}{'='*65}{RESET}\n")

    train = sorted(s for s in scenes if s not in TEST_SCENES)
    test  = sorted(s for s in scenes if s in TEST_SCENES)
    print(f"{BOLD}Scenes:{RESET} {len(scenes)} total  |  {len(train)} train  |  {len(test)} test\n")

    # ── Main table ─────────────────────────────────────────────────────────────
    header = f"{'Scene':<8} {'Files':>6} {'EV missing':>10} {'EV values':<45} {'LDR':>5} {'Split':>7}"
    print(BOLD + header + RESET)
    print("-" * len(header))

    issues = []
    for scene in sorted(scenes):
        info    = scenes[scene]
        n       = info["n_files"]
        miss    = info["ev_missing"]
        has_ldr = scene in ldr_scenes
        split   = "TEST" if scene in TEST_SCENES else "train"

        n_col   = (GREEN if n == 9   else YELLOW) + str(n).rjust(6)     + RESET
        m_col   = (RED   if miss > 0 else GREEN)  + str(miss).rjust(10) + RESET
        ldr_col = (GREEN if has_ldr  else RED)    + ("✓" if has_ldr else "✗").rjust(5) + RESET
        ev_str  = str(info["ev_values"]) if info["ev_values"] else f"{YELLOW}n/a{RESET}"

        print(f"{scene:<8} {n_col} {m_col} {ev_str:<45} {ldr_col} {split:>7}")

        if n != 9:      issues.append(f"{scene}: expected 9 files, got {n}")
        if miss > 0:    issues.append(f"{scene}: {miss} file(s) missing EV metadata")
        if not has_ldr: issues.append(f"{scene}: no matching LDR file")

    orphan = [s for s in ldr_scenes if s not in scenes]
    if orphan:
        issues.append(f"LDR without Bracketed counterpart: {orphan}")

    print()
    if issues:
        print(f"{YELLOW}{BOLD}⚠ Issues:{RESET}")
        for i in issues:
            print(f"  {YELLOW}{i}{RESET}")
    else:
        print(f"{GREEN}{BOLD}✓ No issues found.{RESET}")

    # ── Missing-EV rescue analysis ─────────────────────────────────────────────
    rescue = analyze_missing_ev_scenes(scenes)
    if rescue:
        print(f"\n{BOLD}{'='*65}{RESET}")
        print(f"{BOLD}  ExposureTime analysis — scenes with missing EV metadata{RESET}")
        print(f"{BOLD}{'='*65}{RESET}\n")
        print(f"{'Scene':<8} {'Avg EV step':>11} {'Step range':>22} {'EV first→last':>15} {'Consistent':>11}  {'Rescue?':>8}")
        print("-" * 82)
        for scene in sorted(rescue):
            r = rescue[scene]
            if r["status"] == "no_et":
                print(f"{scene:<8}  {RED}no ExposureTime in any file{RESET}")
                continue
            step_range = f"[{r['min_step']:.3f} – {r['max_step']:.3f}]"
            ev_range   = f"{r['ev_first_est']:+.2f} → {r['ev_last_est']:+.2f}"
            cons_col   = (GREEN + "✓ yes" if r["consistent"] else RED + "✗ no") + RESET
            # rescuable if consistent AND first≈-2.7 AND last≈+2.7
            rescuable  = (r["consistent"]
                          and abs(r["ev_first_est"] - (-2.7)) < 0.4
                          and abs(r["ev_last_est"]  -   2.7)  < 0.4)
            resc_col   = (GREEN + "✓ yes" if rescuable else YELLOW + "? maybe" if r["consistent"] else RED + "✗ no") + RESET
            print(f"{scene:<8}  {r['avg_step']:>10.3f}  {step_range:>22}  {ev_range:>15}  {cons_col:>11}  {resc_col:>8}")

        print(f"\n{BOLD}Detail — sorted files and ET diffs per scene:{RESET}\n")
        for scene in sorted(rescue):
            r = rescue[scene]
            if r["status"] == "no_et":
                continue
            split = "TEST" if scene in TEST_SCENES else "train"
            print(f"  {BOLD}{scene}{RESET} ({split})")
            for i, (fname, et) in enumerate(zip(r["sorted_files"], r["ets"])):
                role = ""
                if i == 0: role = f"  ← {CYAN}underexposed (est. EV {r['ev_first_est']:+.2f}){RESET}"
                if i == 4: role = f"  ← {CYAN}normal (EV  0.0){RESET}"
                if i == 8: role = f"  ← {CYAN}overexposed  (est. EV {r['ev_last_est']:+.2f}){RESET}"
                diff_str = f"  Δ{r['diffs'][i-1]:+.3f} EV" if i > 0 else ""
                print(f"    [{i}] {fname:<25}  ET={et:.6f}s{diff_str}{role}")
            print()

    # ── Summary: usable scenes ─────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  Usability summary{RESET}")
    print(f"{BOLD}{'='*65}{RESET}\n")

    usable_ev    = []   # EV metadata confirms -2.7, 0.0, +2.7
    usable_et    = []   # rescued via ExposureTime
    unusable     = []

    for scene, info in scenes.items():
        evs = info["ev_values"]
        if -2.7 in evs and 0.0 in evs and 2.7 in evs:
            usable_ev.append(scene)
        elif scene in rescue and rescue[scene].get("status") == "ok":
            r = rescue[scene]
            rescuable = (r["consistent"]
                         and abs(r["ev_first_est"] - (-2.7)) < 0.4
                         and abs(r["ev_last_est"]  -   2.7)  < 0.4)
            if rescuable:
                usable_et.append(scene)
            else:
                unusable.append(scene)
        else:
            unusable.append(scene)

    print(f"{GREEN}✓ Usable via EV metadata   ({len(usable_ev):2d}):{RESET} {', '.join(sorted(usable_ev))}")
    print(f"{CYAN}~ Usable via ExposureTime  ({len(usable_et):2d}):{RESET} {', '.join(sorted(usable_et))}")
    print(f"{RED}✗ Unusable                 ({len(unusable):2d}):{RESET} {', '.join(sorted(unusable))}")

    train_usable = [s for s in usable_ev + usable_et if s not in TEST_SCENES]
    test_usable  = [s for s in usable_ev + usable_et if s in TEST_SCENES]
    print(f"\n  → {BOLD}{len(train_usable)} train scenes{RESET} and {BOLD}{len(test_usable)} test scenes{RESET} available for the model.\n")

    print(f"{BOLD}{'='*65}{RESET}\n")


def run():
    scenes     = scan_bracketed()
    ldr_scenes = scan_ldr()
    print_report(scenes, ldr_scenes)