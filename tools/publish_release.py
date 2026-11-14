#!/usr/bin/env python3
"""Creates a GitHub release of ravencaio/tlou-remastered-pc and uploads the package to it.

  python tools/publish_release.py v0.2.4 <full commit SHA> tools/release-notes/v0.2.4.md \
      dist/TLOU-Remastered-windows.zip

The token comes from Git's credential manager (`git credential fill`); gh is not needed. The
commit must already be on GitHub, and it must be the FULL 40-character SHA (a short one gets
HTTP 422). The asset must be named TLOU-Remastered-windows.zip: the launcher's updater
(launcher/updater.py) looks for that name and its SHA-256 digest. See tools/RELEASING.md.
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "ravencaio/tlou-remastered-pc"


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    tag, sha, notes, zip_path = sys.argv[1:5]
    if len(sha) != 40:
        raise SystemExit("give the full 40-character commit SHA (git rev-parse HEAD)")
    cred = subprocess.run(["git", "credential", "fill"],
                          input="protocol=https\nhost=github.com\n\n",
                          capture_output=True, text=True, check=True).stdout
    token = dict(line.split("=", 1) for line in cred.splitlines() if "=" in line)["password"]
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "release-script"}

    def call(url, data, content_type="application/json"):
        req = urllib.request.Request(url, data=data, headers={**headers,
                                                              "Content-Type": content_type})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    body = Path(notes).read_text(encoding="utf-8")
    release = call(f"https://api.github.com/repos/{REPO}/releases",
                   json.dumps({"tag_name": tag, "target_commitish": sha,
                               "name": f"Release {tag}", "body": body,
                               "draft": False, "prerelease": False}).encode())
    print("release", release["id"], release["html_url"])
    data = Path(zip_path).read_bytes()
    asset = call(f"https://uploads.github.com/repos/{REPO}/releases/{release['id']}/assets"
                 f"?name={Path(zip_path).name}", data, "application/zip")
    print("asset", asset["id"], asset["size"], asset.get("digest"), asset["browser_download_url"])


if __name__ == "__main__":
    main()
