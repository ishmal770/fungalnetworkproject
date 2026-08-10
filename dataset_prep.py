import os #file reading and writing
import hashlib #content hashing, used to catch duplicate images
import time #backoff between rate-limited API calls
import requests #used to extract web and api data (here, ask iNaturalist and their api for observations and getting the image files)
import pandas as pd #make dataframes
from PIL import Image #used only to confirm a downloaded file actually decodes

INAT_URL = "https://api.inaturalist.org/v1/observations"
COMMONS_URL = "https://commons.wikimedia.org/w/api.php"

USER_AGENT = {"User-Agent": "EcoScout/1.0 (student science project; "
                            "https://github.com/ishmal770/fungalnetworkproject)"}
FUNGI_TAXON = 47170    # Fungi
PLANTAE_TAXON = 47126  # Plantae


def _fetch_inat(taxon_id, cls, n):
    """Page through iNaturalist observations for one taxon.

    The API caps per_page at 200, so anything larger has to be paged. Every
    request pins taxon_id and quality_grade=research — an unrecognised or
    misspelled param is silently ignored by the API, which means a typo here
    does not error, it just hands back unfiltered observations.
    """
    records, page = [], 1
    while len(records) < n:
        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "quality_grade": "research",  # community-confirmed ID, not someone's guess
            "per_page": min(200, n - len(records)),
            "page": page,
            "order_by": "id",  # stable ordering, so fungi and plant feeds don't interleave
        }
        resp = requests.get(INAT_URL, params=params, timeout=30).json()
        results = resp.get("results", [])
        if not results:
            break
        for obs in results:
            # double-check the API honoured the filter instead of trusting it
            ancestry = obs.get("taxon", {}).get("ancestor_ids", []) or []
            if taxon_id not in ancestry and obs.get("taxon", {}).get("id") != taxon_id:
                continue
            if obs["photos"]:
                image_url = obs["photos"][0]["url"].replace("square", "medium")
                records.append({"image_url": image_url, "class": cls})
        page += 1
    return records[:n]


def fetch_inat_fungi(n=400):
    return _fetch_inat(FUNGI_TAXON, "fungal_fruiting_body", n)


def fetch_inat_nonfungi(n=400):
    
    return _fetch_inat(PLANTAE_TAXON, "none", n)



BACKGROUND_QUERIES = [
    "soil texture ground", "bare earth field ground", "dry cracked mud",
    "sky clouds", "overcast sky", "blue sky",
    "gravel texture", "sand texture", "asphalt road surface", "concrete pavement",
    "mowed lawn grass", "dry grass field", "bark mulch garden",
    "leaf litter forest floor", "moss covered rock", "tree bark texture",
    "wood chips ground", "pine needles ground", "pebbles stones ground",
    "ploughed field soil", "meadow grass ground", "footpath dirt track",
]


_NOT_BACKGROUND = (
    "fungus", "fungi", "mushroom", "toadstool", "agaric", "bolet", "amanita",
    "polypore", "mycena", "russula", "lichen", "mycel",
    "flower", "blossom", "bloom", "leaf ", "leaves", "portrait", "man ", "woman",
    "church", "cathedral", "castle", "building", "statue", "bird", "beetle",
    "moth", "butterfly", "spider", "snail", "logo", "map", "diagram", "chart",
)


def _commons_search(query, limit):
    """One throttled Commons search. Returns (title, thumb_url) pairs."""
    params = {
        "action": "query", "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": "6",          # File: namespace only
        "gsrlimit": str(limit),
        "prop": "imageinfo", "iiprop": "url",
        "iiurlwidth": "600",          # ask for a 600px thumb, not the 20MB original
        "format": "json",
    }
    resp = requests.get(COMMONS_URL, params=params, headers=USER_AGENT, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    pages = resp.json().get("query", {}).get("pages", {})
    out = []
    for pg in pages.values():
        info = (pg.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        if url:
            out.append((pg.get("title", ""), url))
    return out


def fetch_commons_backgrounds(per_query=40, pause=2.0):
    """Wikimedia Commons for ground and sky imagery — no API key, stable under a
    polite request rate. Precision is imperfect, but the label is only "a frame
    with no fungus in it", so an off-target park bench is still a valid negative.
    Anything whose title suggests an organism is dropped by _NOT_BACKGROUND."""
    seen, records, rejected = set(), [], 0
    for q in BACKGROUND_QUERIES:
        try:
            hits = _commons_search(q, per_query)
        except Exception as e:
            print(f"skip background query {q!r}: {e}")
            time.sleep(pause * 4)
            continue
        kept = 0
        for title, url in hits:
            low = title.lower()
            if any(bad in low for bad in _NOT_BACKGROUND):
                rejected += 1
                continue
            if url in seen:
                continue
            seen.add(url)
            records.append({"image_url": url, "class": "background"})
            kept += 1
        print(f"  {q!r}: {kept}/{len(hits)} kept ({len(records)} total)")
        time.sleep(pause)
    print(f"background search: {len(records)} candidates, {rejected} rejected by title filter")
    return records


def _fetch_bytes(url, session, tries=4, pause=2.0):
    """Download one image, backing off on 429. A rate-limited response is an HTML
    error page with a 200-ish body — writing that to a .jpg produces a file that
    only fails much later, mid-training, so treat a non-image content-type as an
    error here where it is still cheap to retry or skip."""
    for attempt in range(tries):
        resp = session.get(url, timeout=20, headers=USER_AGENT)
        if resp.status_code == 429 or not resp.headers.get("Content-Type", "").startswith("image/"):
            if attempt == tries - 1:
                raise RuntimeError(f"HTTP {resp.status_code} {resp.headers.get('Content-Type')}")
            time.sleep(pause * (2 ** attempt))
            continue
        return resp.content
    raise RuntimeError("exhausted retries")


def download_url_images(records, out_dir, prefix, pause=0.4):
    os.makedirs(out_dir, exist_ok=True)
    out = []
    session = requests.Session()
    for i, r in enumerate(records):
        fname = f"{prefix}_{i:04d}.jpg"
        path = os.path.join(out_dir, fname)
        # already on disk from a previous run — re-running the pipeline to add one
        # new source should not re-download the other 800 images
        if os.path.exists(path) and os.path.getsize(path) > 0:
            out.append({"image_path": path, "class": r["class"], "source": prefix})
            continue
        try:
            img_data = _fetch_bytes(r["image_url"], session)
            with open(path, "wb") as f:
                f.write(img_data)
            Image.open(path).convert("RGB")  # confirm it really decodes
            out.append({"image_path": path, "class": r["class"], "source": prefix})
        except Exception as e:
            print(f"skip {i}: {e}")
            if os.path.exists(path):
                os.remove(path)
        time.sleep(pause)  # polite, and keeps the CDN from rate-limiting the run
    return out


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def label_folder_dataset(root_dir, prefix, keep_classes=None, per_class_cap=None):
    """Collect labelled leaf images out of a PlantVillage/PlantDoc-style tree.

    These datasets ship as root/<split>/<class>/*.jpg, so the class folders sit
    one level below where a naive listdir looks. Walk instead of listing, and
    match keep_classes against the class folder anywhere in the path.
    """
    records = []
    counts = {}
    if not os.path.isdir(root_dir):
        print(f"warn: {root_dir} not found, skipping")
        return records

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        label_folder = os.path.basename(dirpath)
        images = [f for f in filenames if f.lower().endswith(IMG_EXTS)]
        if not images:
            continue
        if keep_classes and label_folder not in keep_classes:
            continue
        cls = "none" if "healthy" in label_folder.lower() else "vegetative_stress"
        for fname in sorted(images):
            if per_class_cap and counts.get(label_folder, 0) >= per_class_cap:
                break
            records.append({
                "image_path": os.path.join(dirpath, fname),
                "class": cls,
                "source": prefix,
            })
            counts[label_folder] = counts.get(label_folder, 0) + 1
    return records


# ---------- weak composite label ----------
def compute_network_score(row):
    w1, w2 = 0.7, 0.3
    fruiting_body_density = 1.0 if row["class"] == "fungal_fruiting_body" else 0.0
    veg_anomaly_score = 1.0 if row["class"] == "vegetative_stress" else 0.0
    return w1 * fruiting_body_density + w2 * veg_anomaly_score


def _file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def dedupe_records(records):
    """Drop byte-identical images, and drop entirely any image that shows up
    under two different labels — a contradictory pair teaches the model nothing
    except to memorise, and if the copies land on opposite sides of the split it
    also leaks test data into training."""
    by_hash = {}
    for r in records:
        h = _file_hash(r["image_path"])
        if h is None:
            continue
        by_hash.setdefault(h, []).append(r)

    kept, conflicts, dups = [], 0, 0
    for group in by_hash.values():
        labels = {r["class"] for r in group}
        if len(labels) > 1:
            conflicts += 1
            continue  # same pixels, two labels -> unusable, drop the whole group
        dups += len(group) - 1
        kept.append(group[0])
    print(f"dedupe: dropped {dups} exact duplicates, {conflicts} cross-label conflict groups")
    return kept


def balance_classes(df, cap=None, seed=42):
    """Cap every class at the size of the smallest one (or `cap`). PlantVillage
    alone is ~7k images against a few hundred iNaturalist ones; left unbalanced
    the model just predicts the majority class and the accuracy looks fine."""
    smallest = df["class"].value_counts().min()
    target = min(smallest, cap) if cap else smallest
    parts = [g.sample(min(len(g), target), random_state=seed)
             for _, g in df.groupby("class", sort=False)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    OUT_DIR = "datasets"
    IMG_DIR = os.path.join(OUT_DIR, "images")

    
    inat_records = download_url_images(fetch_inat_fungi(400), IMG_DIR, "inat")
    inat_nonfungi_records = download_url_images(fetch_inat_nonfungi(400), IMG_DIR, "inat_none")

    
    bg_records = download_url_images(fetch_commons_backgrounds(), IMG_DIR, "bg")

    # 2. PlantVillage subset — folder names as they appear on disk (root/<split>/<class>)
    PV_CLASSES = ["Tomato___healthy", "Tomato___Early_blight",
                  "Apple___healthy", "Apple___Apple_scab",
                  "Potato___healthy", "Potato___Late_blight"]
    pv_records = label_folder_dataset("datasets/PlantVillage", "plantvillage",
                                      keep_classes=PV_CLASSES, per_class_cap=150)

    # 3. PlantDoc subset
    PD_CLASSES = ["Tomato_leaf", "Tomato_Early_blight_leaf", "Tomato_leaf_bacterial_spot",
                  "Tomato_Septoria_leaf_spot", "Tomato_leaf_late_blight", "Tomato_mold_leaf"]
    pd_records = label_folder_dataset("datasets/plantdoc", "plantdoc",
                                      keep_classes=PD_CLASSES, per_class_cap=150)

    all_records = inat_records + inat_nonfungi_records + bg_records + pv_records + pd_records
    all_records = dedupe_records(all_records)

    df = pd.DataFrame(all_records)
    df = balance_classes(df)
    df["network_score"] = df.apply(compute_network_score, axis=1)
    df.to_csv(os.path.join(OUT_DIR, "labels.csv"), index=False)
    print(f"Wrote {len(df)} rows to {OUT_DIR}/labels.csv")
    print(df["class"].value_counts())
    # if any class comes from exactly one source, the model can hit high accuracy by
    # recognising the dataset rather than the subject — print it so it can't hide
    print("\nclass x source (a class confined to one column is a shortcut waiting to happen):")
    print(pd.crosstab(df["class"], df["source"]).to_string())

    # train/test split, stratified by class so rare classes don't disappear from one side
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=42)
    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)
    print(f"Split: {len(train_df)} train / {len(test_df)} test")