"""
Helper для заливки results folder в приватный HuggingFace Dataset repo.

Использование:
    # На локальной машине после scp с Vast:
    python3 src/upload_to_hf.py results/run_2026-05-28_qwen3.5-2b/

    # Кастомный repo / sub-path:
    python3 src/upload_to_hf.py results/run_X/ \\
        --repo-id olyamasaeva/qwen-bias-experiments \\
        --path-in-repo run_2026-05-28_qwen3.5-2b

    # На Vast напрямую (HF token из env):
    python3 src/upload_to_hf.py /workspace/results/run_2026-05-28_qwen3.5-2b/

Токен берётся:
    1. из --token аргумента (если задан)
    2. иначе из env: HF_TOKEN / HUGGING_FACE_HUB_TOKEN
    3. иначе из ~/.cache/huggingface/token (после huggingface-cli login)

Repo создаётся автоматически (private=True по умолчанию).
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO = "olyamasaeva/qwen-bias-experiments"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Путь к локальной папке с результатами (например results/run_2026-05-28_...)")
    parser.add_argument("--repo-id", default=DEFAULT_REPO,
                        help=f"HF Dataset repo id (default: {DEFAULT_REPO})")
    parser.add_argument("--path-in-repo", default=None,
                        help="Sub-path внутри repo. По умолчанию = имя локальной папки.")
    parser.add_argument("--token", default=None,
                        help="HF token. Если не задан — берётся из env HF_TOKEN или ~/.cache/huggingface/token.")
    parser.add_argument("--public", action="store_true",
                        help="Создать public repo (по умолчанию private).")
    parser.add_argument("--commit-message", default=None,
                        help="Custom commit message.")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"❌ {folder} не существует или не папка")

    path_in_repo = args.path_in_repo or folder.name
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    api = HfApi(token=token)   # если token=None, использует кэш HF

    print(f"[1/3] ensure repo: {args.repo_id} (private={not args.public})")
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)

    commit_msg = args.commit_message or f"upload {folder.name}"
    print(f"[2/3] upload {folder} → {path_in_repo}/")
    print(f"      files: {sorted(p.name for p in folder.iterdir())}")

    api.upload_folder(
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=commit_msg,
    )

    print(f"[3/3] ✓ done")
    print(f"      https://huggingface.co/datasets/{args.repo_id}/tree/main/{path_in_repo}")


if __name__ == "__main__":
    main()
