import os #file reading and writing
import requests #used to extract web and api data (here, ask iNaturalist and their api for observations and getting the image files)
import pandas as pd #make dataframes

def fetch_inat_fungi(n=150): #gets fungi observations from Inaturalist (150 observations)
    url = "https://api.inaturalist.org/v1/observations" # get url of inaturalist to get data
    params = {
        "taxion_id": 47170, #gets observations from 47170 taxonomic group (or fungis)
        "photos": "true", #get photos
        "per_page": n, # in a page, get 150 of those observations
        "order_by": "created_at", #order by creation date
    } 

    resp = requests.get(url, params=params).json() #we send a get (to get something from) the api, or inaturalist, and go through it with these parameters, in which it sorts it through
    records = [] 
    for obs in resp["results"]: #for observation in the request results
        if obs["photos"]:  #if the observation photo, we get the list of photos in the obsevation, 
            image_url = obs["photos"][0]["url"].replace("square", "medium")
            # we get the first photo, and then gets the url of it; and replaces names (square) with medium so we get medium-sized images.
            records.append({"image_url": image_url, "class": "fungal_fruiting_body"})
            # then append the image url and class name into the records list
    return records


def download_url_images(records, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for i, r in enumerate(records):
        try:
            image_data = requests.get(r["image_url"], timeout=10).content
            fname = f"{prefix}_{i:04d}.jpg"
            path = os.path.join(out_dir, fname)
            with open(path, "wb") as f:
                f.write(image_data)
            out.append({"image_path": path, "class": r["class"]})
        except Exception as e:
            print(f"skip {i}: {e}")
    return out

def label_folder_dataset(root_dir, prefix, keep_classes=None):
    records = []
    for label_folder in os.listdir(root_dir):
        if keep_classes and label_folder not in keep_classes:
            continue
        cls = "none" if "healthy" in label_folder.lower() else "vegetative_stress"
        folder_path = os.path.join(root_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            records.append({
                "image_path": os.path.join(folder_path, fname),
                "class": cls,
                "source": prefix
            })
    return records

def compute_network_score(row):
    w1, w2 = 0.7, 0.3
    fruiting_body_density = 1.0 if row["class"] == "fungal_fruiting_body" else 0.0
    veg_anamoly_score = 1.0 if row["class"] == "vegetative_stress" else 0.0
    return w1 * fruiting_body_density + w2 *veg_anamoly_score

if __name__ == "__main__":
    OUT_DIR = "datasets"
    IMG_DIR = os.path.join(OUT_DIR, "images")

    inat_raw = fetch_inat_fungi(150)
    inat_records = download_url_images(inat_raw, IMG_DIR, "inant")

    PV_CLASSES = [
        "Tomato_healthy", 
        "Tomato_Early_blight", 
        "Apple_healthy", 
        "Apple_scab"
    ]

    pv_records = label_folder_dataset("datasets/PlantVillage", "PlantVillage", keep_classes = PV_CLASSES)

    PD_CLASSES = [
        "Tomato_leaf_healthy", 
        "Tomato_leaf_bacterial_spot", 
        "Apple_leaf_healthy", 
        "Apple_Scab_Leaf"
    ]

    pd_records = label_folder_dataset("datasets/plantdoc", "plantdoc", keep_classes=PD_CLASSES)
 
    all_records = inat_records + pv_records + pd_records
    df = pd.DataFrame(all_records)
    df["network_score"] = df.apply(compute_network_score, axis=1)
    df.to_csv(os.path.join(OUT_DIR, "labels.csv"), index=False)
    print(f"Wrote {len(df)} rows to {OUT_DIR}/labels.csv")


    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=42)
    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)
    print(f"Split: {len(train_df)} train / {len(test_df)} test")