# Publishing a release

How a release of the port is made, step by step. Everything runs from Git Bash in the repository
root on the Windows machine that builds the port. Work happens on the local branch `tlou-port`;
GitHub's `main` is pushed from it.

## 1. Check the state

```bash
git status --short | grep -v '^??'      # only expected changes; never commit gow3_pc/gpu/third_party/fsr-vulkan
git fetch origin && git log --oneline origin/main..tlou-port   # what the release adds
cat tools/VERSION                       # the previous version
```

- The `fsr-vulkan` submodule always shows as modified (`build.sh` patches it): leave it out.
- Untracked files at the root (`Teste-*.cmd`, logs, `bug_rsc/`, zips...) are the maintainer's
  working material: never add or delete them.

## 2. Write the release notes

Create `tools/release-notes/v<version>.md`, copying the layout of the previous one:

- Title line `**The Last of Us Remastered: PC port** <version>`, a one-paragraph summary, the
  "No game files are included" quote and the Discord line.
- `## Fixes and improvements`: one bullet per change, written for players, with issue links
  (`[#55](https://github.com/ravencaio/tlou-remastered-pc/issues/55)`). Say what is not tested yet.
- `## Update`, `## New install`, `## Known issues`: copy them and update the known issues.
- Credit issues only by number. No thank-you lines.

**Show the notes to the maintainer and wait for approval before publishing.**

## 3. Version commit and push

```bash
echo <version> > tools/VERSION
git add tools/VERSION tools/release-notes/v<version>.md
git commit -m "The Last of Us Remastered: release <version>"   # plus the Co-Authored-By line
git push origin tlou-port:main
git rev-parse HEAD        # the full SHA, needed in step 5
```

- **Never `git push --tags`.** The local tags `v0.2.0`, `v0.2.2`, `v0.2.3`... are the upstream
  Uncharted port's and clash with this repository's. The release API creates the tag on GitHub.

## 4. Build the package

```bash
mv dist/TLOU-Remastered-windows.zip dist/TLOU-Remastered-windows-v<previous>.zip   # keep the old one
bash tools/package_windows.sh > dist/package-<version>.log 2>&1; echo "exit $?"
ls -la dist/TLOU-Remastered-windows.zip
```

- It rebuilds the runtime (PGO + LTO, several minutes), the starters (`launcher/build-starters.sh`)
  and the Python runtime, then zips `dist/TLOU-Remastered-windows.zip` (~78 MB).
- The game must be closed: a running `gow3-probe.exe` cannot be replaced.
- `VERSION` in the zip comes from `tools/VERSION`; the launcher's updater compares it with the
  latest release's tag.

## 5. Publish

```bash
python tools/publish_release.py v<version> <full SHA> tools/release-notes/v<version>.md \
    dist/TLOU-Remastered-windows.zip
```

- `gh` is not installed; the script uses the GitHub REST API with the token from
  `git credential fill`.
- Tag `v<version>`, title `Release v<version>`, asset `TLOU-Remastered-windows.zip`. The asset
  name must stay exactly that: `launcher/updater.py` looks it up with its `sha256:` digest.
- A short SHA gets HTTP 422: use `git rev-parse HEAD`.

## 6. Check

```bash
curl -s https://api.github.com/repos/ravencaio/tlou-remastered-pc/releases/latest | \
    python -c "import json,sys; r=json.load(sys.stdin); print(r['tag_name'], r['target_commitish'], [(a['name'], a['size'], a.get('digest')) for a in r['assets']])"
```

The tag, the commit, the asset name, its size (same as the local zip) and a `sha256:` digest
must all be there. Give the maintainer the release link.
