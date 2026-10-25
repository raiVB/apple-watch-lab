"""
One-time (or occasional) history loader. Run this on your COMPUTER or Google Colab, not on Render.
It streams the big Health Auto Export JSON file(s) straight into MongoDB, so Render's 512 MB
limit is never involved.

    pip install "pymongo[srv]==4.6.2" certifi "ijson>=3.2"
    export MONGO_URI="mongodb+srv://..."          (Windows PowerShell: $env:MONGO_URI="mongodb+srv://...")
    python backfill.py path/to/export.json --days 120 --wipe

  paths   one or more .json or .xml files, or a folder that contains them (unzip Apple export first)
  --days  only keep the last N days (default 120; this keeps MongoDB's free 512 MB safe)
  --wipe  empty the old collections first. Needed once, because the old rows were raw samples
          and the new ones are 5-minute bins.
"""
import argparse
import glob
import os
import sys
import time
from datetime import datetime, timedelta

import certifi
from pymongo import MongoClient

import ingest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--wipe", action="store_true")
    a = ap.parse_args()

    uri = os.environ.get("MONGO_URI")
    if not uri:
        sys.exit("Set the MONGO_URI environment variable first.")
    db = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=15000).biolab

    files = []
    for p in a.paths:
        files += (sorted(glob.glob(os.path.join(p, "*.json"))) + sorted(glob.glob(os.path.join(p, "*.xml")))) if os.path.isdir(p) else [p]
    if not files:
        sys.exit("No .json files found.")

    if a.wipe:
        for c in ("records", "sleep", "sleep_raw"):
            db[c].drop()
        print("Old collections wiped.")
    ingest.ensure_indexes(db)

    keep_from = (datetime.now() - timedelta(days=a.days)).strftime("%Y-%m-%d")
    ing = ingest.Ingestor(keep_from=keep_from)  # one Ingestor for all files, so 5-min bins that span files stay exact
    for path in files:
        t0 = time.time()
        ingest.ingest_file(path, ing)
        print(f"read {os.path.basename(path)}: {ing.items:,} records so far, {len(ing.bins):,} bins ({time.time() - t0:.0f}s)")

    print("Writing to MongoDB...")
    print(ingest.write(db, ing))
    print("Done. Open /stats on your Render app to confirm.")


if __name__ == "__main__":
    main()
