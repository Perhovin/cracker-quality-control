"""Storage and table building for the user's own photos.

Layout: MY_DATA/<name>/{ok,ng,holdout_ok,holdout_ng}/ with files named
<item:04d>_<shot:02d>.jpg. The "group" of a photo is the item id, the
text before the first "_". All shots of one item always land in the
same dataset split, so the model never trains and is never tested on
photos of the very same physical object.
"""

import hashlib
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

import paths

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")

# the standard MVTec category names; a custom dataset may not reuse one
MVTEC_NAMES = {
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
}

FOLDERS = {
    (0, False): "ok",
    (1, False): "ng",
    (0, True): "holdout_ok",
    (1, True): "holdout_ng",
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_HINT = "set the phone camera to JPEG / Most Compatible"
MAX_SIDE = 1600
SEED = 42


def valid_name(name):
    """True for a usable dataset name (also checked against MVTec)."""
    if not NAME_RE.match(name or ""):
        return False
    if name in MVTEC_NAMES:
        return False
    root = paths.get_mvtec_root()
    if root.is_dir() and (root / name).is_dir():
        return False
    return True


def name_error(name):
    """Why a name was rejected, for messages."""
    if not NAME_RE.match(name or ""):
        return ("use 2-31 characters: lowercase letters, digits and "
                "'_', starting with a letter")
    return "the name is already used by an MVTec category"


def _folder(name, label, holdout):
    return paths.MY_DATA / name / FOLDERS[(label, holdout)]


def _items(folder):
    """Item ids present in one folder."""
    if not folder.is_dir():
        return set()
    ids = set()
    for p in folder.iterdir():
        if p.is_file():
            ids.add(p.stem.split("_", 1)[0])
    return ids


def list_custom_datasets():
    """Folders under my_data that look like a dataset."""
    if not paths.MY_DATA.is_dir():
        return []
    return sorted(
        p.name for p in paths.MY_DATA.iterdir()
        if p.is_dir() and NAME_RE.match(p.name)
    )


def next_item_id(name):
    """Lowest unused item id across all four folders (1-based)."""
    base = paths.MY_DATA / name
    used = set()
    for f in FOLDERS.values():
        used |= _items(base / f)
    n = 1
    while f"{n:04d}" in used:
        n += 1
    return n


def next_shot(name, item_id):
    """Next shot number for an item across all four folders."""
    base = paths.MY_DATA / name
    prefix = f"{item_id:04d}_"
    n = 0
    for f in FOLDERS.values():
        d = base / f
        if d.is_dir():
            for p in d.iterdir():
                if p.stem.startswith(prefix):
                    try:
                        n = max(n, int(p.stem.split("_", 1)[1]))
                    except ValueError:
                        pass
    return n + 1


def save_array_photo(name, bgr_frame, label, holdout, item_id):
    """Write one camera frame (BGR numpy array) as JPEG quality 95."""
    import cv2
    d = _folder(name, label, holdout)
    d.mkdir(parents=True, exist_ok=True)
    shot = next_shot(name, item_id)
    path = d / f"{item_id:04d}_{shot:02d}.jpg"
    cv2.imwrite(
        str(path), bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def delete_photo(path):
    """Delete one photo file; returns True if something was removed."""
    path = Path(path)
    if path.is_file():
        path.unlink()
        return True
    return False


def counts(name):
    """{folder: (items, photos)} for each of the four folders."""
    base = paths.MY_DATA / name
    out = {}
    for (label, holdout), f in FOLDERS.items():
        d = base / f
        photos = [p for p in d.iterdir() if p.is_file()] \
            if d.is_dir() else []
        out[f] = (len(_items(d)), len(photos))
    return out


def _natural_key(path):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", path.name)]


def import_folder(name, src, label, holdout, per_item, progress_cb=None):
    """Copy photos from a folder into a dataset, N files per item.

    Every image is opened with Pillow, EXIF orientation applied,
    converted to RGB, downscaled to <=1600 px on the long side and
    saved as JPEG quality 95. Source files are never modified.
    """
    src = Path(src)
    files = sorted(
        (p for p in src.iterdir() if p.is_file()), key=_natural_key)
    good = [p for p in files if p.suffix.lower() in IMG_EXT]
    skipped = [p.name for p in files if p.suffix.lower() not in IMG_EXT]
    imported = 0
    item_id = next_item_id(name)
    shot = 0
    errors = []
    for i, p in enumerate(good):
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                if max(w, h) > MAX_SIDE:
                    s = MAX_SIDE / max(w, h)
                    im = im.resize(
                        (max(1, round(w * s)), max(1, round(h * s))),
                        Image.LANCZOS)
                d = _folder(name, label, holdout)
                d.mkdir(parents=True, exist_ok=True)
                shot += 1
                im.save(
                    d / f"{item_id:04d}_{shot:02d}.jpg",
                    "JPEG", quality=95)
            imported += 1
            if imported % max(1, per_item) == 0:
                item_id = next_item_id(name)
                shot = 0
        except Exception as e:
            errors.append(f"{p.name}: {e}")
        if progress_cb:
            progress_cb(i + 1, len(good))
    return {
        "imported": imported,
        "items": len(_items(_folder(name, label, holdout))),
        "skipped": skipped,
        "errors": errors,
        "hint": SKIP_HINT if skipped else "",
    }


def _collect(name):
    """All photos with (item, label, holdout, path); validates layout."""
    base = paths.MY_DATA / name
    rows = []
    seen = {}          # item id -> (label, holdout) of first photo
    for (label, holdout), f in FOLDERS.items():
        d = base / f
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file() or "_" not in p.stem:
                continue
            item = p.stem.split("_", 1)[0]
            if item in seen and seen[item] != (label, holdout):
                raise ValueError(
                    f"item {item} has photos in more than one folder "
                    f"({p.parent.name} and others) - an item must be "
                    f"either OK or NG, normal or final-exam, not both")
            seen[item] = (label, holdout)
            rows.append({
                "path": str(p.resolve()),
                "label": label,
                "holdout": holdout,
                "group": item,
            })
    return rows


def build_table(name):
    """Create dataset_<name>.csv from the my_data folders.

    Splits by ITEM, never by photo. Without holdout items each class
    is split ~70/15/15; with holdout items, all of them become the
    test split and the rest is split into train/val.
    """
    if not valid_name(name):
        raise ValueError(f"invalid dataset name '{name}': {name_error(name)}")
    rows = _collect(name)
    if not rows:
        raise ValueError(f"dataset '{name}' has no photos yet")

    items = {}
    for r in rows:
        items.setdefault(r["group"], {"label": r["label"],
                                      "holdout": r["holdout"],
                                      "photos": []})
        items[r["group"]]["photos"].append(r)

    for label in (0, 1):
        if not any(i["label"] == label for i in items.values()):
            raise ValueError(
                f"dataset '{name}' has no "
                f"{'OK' if label == 0 else 'NG'} items - photograph "
                f"some first")

    # identical file content in two different items is suspicious
    hashes = {}
    for r in rows:
        h = hashlib.md5(Path(r["path"]).read_bytes()).hexdigest()
        if h in hashes and hashes[h][0] != r["group"]:
            raise ValueError(
                f"identical file content in items {hashes[h][0]} and "
                f"{r['group']} ({Path(hashes[h][1]).name} and "
                f"{Path(r['path']).name}) - each item must be a "
                f"different physical object")
        hashes[h] = (r["group"], r["path"])

    warnings = []

    # image checks: readable, aspect ratio, size
    ratios, bad, min_short = Counter(), [], None
    for r in rows:
        try:
            with Image.open(r["path"]) as im:
                w, h = im.size
        except Exception:
            bad.append(Path(r["path"]).name)
            continue
        ratios[round(w / h, 2)] += 1
        s = min(w, h)
        if min_short is None or s < min_short:
            min_short = s
    if bad:
        raise ValueError("could not read these photos: " + ", ".join(bad))
    if len(ratios) > 1:
        warnings.append(
            "mixed aspect ratios (" + ", ".join(
                f"{k} x{v}" for k, v in sorted(ratios.items())) +
            ") - shoot with one fixed camera orientation")
    if min_short is not None and min_short < 256:
        warnings.append(
            f"shortest photo side is {min_short}px (<256) - the model "
            f"sees a 256x256 square, small photos lose detail")

    def split_items(ids, n_test, n_val):
        ids = sorted(ids)
        random.Random(SEED).shuffle(ids)
        return ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]

    for r in rows:
        r["split"] = None
    any_hold = any(i["holdout"] for i in items.values())
    for label in (0, 1):
        all_ids = [g for g, i in items.items() if i["label"] == label]
        if any_hold and not any(items[g]["holdout"] for g in all_ids):
            raise ValueError(
                f"the final-exam set needs both OK and NG items - "
                f"no final-exam {'OK' if label == 0 else 'NG'} items yet")
        hold_ids = [g for g in all_ids if items[g]["holdout"]]
        norm_ids = [g for g in all_ids if not items[g]["holdout"]]
        cls = "OK" if label == 0 else "NG"
        if hold_ids:
            if len(norm_ids) < 2:
                raise ValueError(
                    f"needs at least 2 normal {cls} items plus the "
                    f"final-exam items (have {len(norm_ids)})")
            for g in hold_ids:
                for r in items[g]["photos"]:
                    r["split"] = "test"
            n_val = max(1, round(0.2 * len(norm_ids)))
            _, val, train = split_items(norm_ids, 0, n_val)
            split_map = {"val": val, "train": train}
        else:
            n = len(all_ids)
            if n < 3:
                raise ValueError(
                    f"needs at least 3 {cls} items to split into "
                    f"train/val/test (have {n})")
            test, val, train = split_items(
                all_ids, max(1, round(0.15 * n)), max(1, round(0.15 * n)))
            split_map = {"test": test, "val": val, "train": train}
        for s, ids in split_map.items():
            for g in ids:
                for r in items[g]["photos"]:
                    r["split"] = s

    n_ok = sum(1 for i in items.values() if i["label"] == 0)
    n_ng = sum(1 for i in items.values() if i["label"] == 1)
    big, small = max(n_ok, n_ng), max(1, min(n_ok, n_ng))
    if big / small > 10:
        warnings.append(
            f"class imbalance {big}:{small} - usable, but aim for a "
            f"more even mix (pos_weight in training compensates partly)")
    for label, cls in ((0, "OK"), (1, "NG")):
        n = sum(1 for r in rows
                if r["split"] == "val" and r["label"] == label)
        if 0 < n < 5:
            warnings.append(
                f"only {n} {cls} photos in validation - the reported "
                f"accuracy will be noisy; add more items")

    df = pd.DataFrame({
        "path": [r["path"] for r in rows],
        "label": [r["label"] for r in rows],
        "defect": ["defect" if r["label"] else "good" for r in rows],
        "split": [r["split"] for r in rows],
        "group": [r["group"] for r in rows],
    })
    out = paths.BASE / f"dataset_{name}.csv"
    df.to_csv(out, index=False)

    report = {
        "csv": str(out),
        "rows": len(df),
        "items": {"ok": n_ok, "ng": n_ng},
        "warnings": warnings,
        "splits": {},
    }
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        report["splits"][s] = {
            "photos": len(sub),
            "ok_photos": int((sub["label"] == 0).sum()),
            "ng_photos": int((sub["label"] == 1).sum()),
            "items": sub["group"].nunique(),
        }
    return report
