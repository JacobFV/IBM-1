"""migrate HuggingFace repos between accounts, verifying before deleting anything.

`move_repo` is refused in both directions here: the source and destination are
separate user accounts, and HuggingFace requires one token with write access to
both.  so this copies rather than moves, and the ordering is the whole point --
**nothing is deleted until the destination has been verified file by file.**

that ordering matters more than it sounds.  `brandonin/ibm-1` holds 52 training
checkpoints and only 3 exist on local disk; a delete-then-recreate would have
destroyed 49 artefacts that exist nowhere else.

two tokens are used deliberately: the source is read with whatever token owns it,
and the destination is written with the token that owns the destination.  the
script refuses to start if either identity is not what was expected.
"""
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download


def api_for(token: str | None) -> HfApi:
    return HfApi(token=token) if token else HfApi()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", nargs="+", required=True,
                    help="src=dst pairs, e.g. brandonin/ibm-1=jacob-valdez/ibm-1")
    ap.add_argument("--src-user", required=True)
    ap.add_argument("--dst-user", required=True)
    ap.add_argument("--cache", default="/tmp/hf_migrate")
    ap.add_argument("--delete-source", action="store_true",
                    help="only honoured after verification passes")
    a = ap.parse_args()

    # SRC_HF_TOKEN, not HF_TOKEN: the latter is picked up implicitly by every
    # HfApi() constructed without an explicit token, which made both the source
    # and the destination authenticate as the source account.  the identity
    # guard below caught that before anything was touched.
    src_token = os.environ.get("SRC_HF_TOKEN")       # owns the source
    dst_api = api_for(None)                          # the CLI's active token
    src_api = api_for(src_token)

    who_src = src_api.whoami()["name"]
    who_dst = dst_api.whoami()["name"]
    print(f"source identity: {who_src}   destination identity: {who_dst}", flush=True)
    if who_src != a.src_user or who_dst != a.dst_user:
        sys.exit(f"identity mismatch: expected {a.src_user}/{a.dst_user}, "
                 f"got {who_src}/{who_dst} -- refusing to touch anything")

    for pair in a.repos:
        src, dst = pair.split("=", 1)
        print(f"\n=== {src} -> {dst} ===", flush=True)

        src_files = {s.rfilename for s in src_api.model_info(src).siblings}
        print(f"  source has {len(src_files)} files", flush=True)

        local = snapshot_download(src, repo_type="model", token=src_token,
                                  cache_dir=a.cache, max_workers=8)
        print(f"  downloaded to {local}", flush=True)

        dst_api.create_repo(dst, repo_type="model", private=False, exist_ok=True)
        dst_api.upload_folder(folder_path=local, repo_id=dst, repo_type="model",
                              commit_message=f"migrated from {src}")
        print("  uploaded", flush=True)

        # ---- verify BEFORE any deletion -------------------------------
        dst_files = {s.rfilename for s in dst_api.model_info(dst).siblings}
        missing = src_files - dst_files
        if missing:
            print(f"  VERIFY FAILED: {len(missing)} files missing at destination")
            for m in sorted(missing)[:8]:
                print(f"    {m}")
            print("  source left untouched")
            continue
        print(f"  verified: all {len(src_files)} files present at {dst}", flush=True)

        if a.delete_source:
            src_api.delete_repo(repo_id=src, repo_type="model")
            print(f"  deleted {src}", flush=True)
        else:
            print(f"  source kept (pass --delete-source to remove it)", flush=True)


if __name__ == "__main__":
    main()
