"""Copy gpxil-read.js into a site that keeps a copy (the Gpxil browser, GpxilStudio), with a record of where it came
from: gpxil-read.source.json beside it, naming this repository, the commit and the file's SHA-256, which the site's
tests check.

    python3 js/copy_to.py DEST_DIR

Run it on a commit of main that is on GitHub, with gpxil-read.js as committed: the record names that commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/comparativechrono/Gpxil-tools"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=HERE, check=True, capture_output=True, text=True).stdout.strip()


def main(dest: str) -> int:
    src = os.path.join(HERE, "gpxil-read.js")
    if git("status", "--porcelain", "--", src):
        print("gpxil-read.js has changes not committed: commit them first, so that the record names them", file=sys.stderr)
        return 1
    commit = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", commit):
        print(f"warning: {commit[:7]} is not on GitHub yet; push it before the copy is published", file=sys.stderr)
    os.makedirs(dest, exist_ok=True)
    shutil.copyfile(src, os.path.join(dest, "gpxil-read.js"))
    data = open(src, "rb").read()
    record = {
        "about": "gpxil-read.js is a copy of gpxiltools' JavaScript reader, unchanged; update it with js/copy_to.py there",
        "repository": REPO,
        "path": "js/gpxil-read.js",
        "commit": commit,
        "sha256": hashlib.sha256(data).hexdigest(),
        "url": f"https://raw.githubusercontent.com/comparativechrono/Gpxil-tools/{commit}/js/gpxil-read.js",
    }
    with open(os.path.join(dest, "gpxil-read.source.json"), "w") as fh:
        json.dump(record, fh, indent=1)
        fh.write("\n")
    print(f"copied gpxil-read.js at {commit[:7]} to {dest}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
