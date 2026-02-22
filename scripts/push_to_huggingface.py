import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import CACHED_MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description="Push model and tokenizer to Hugging Face Hub")
    parser.add_argument(
        "--repo-name",
        required=True,
        help="Hugging Face repo: 'repo' or 'username/repo'. Default username from HF_USERNAME.",
    )
    parser.add_argument(
        "--local-dir",
        default=CACHED_MODEL_DIR,
        help="Local directory containing saved model and config (default: {})".format(CACHED_MODEL_DIR),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repo (default: public for Task 7).",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN (e.g. from https://huggingface.co/settings/tokens) to push.")
        sys.exit(1)

    repo_id = args.repo_name
    if "/" not in repo_id and os.environ.get("HF_USERNAME"):
        repo_id = "{}/{}".format(os.environ.get("HF_USERNAME"), repo_id)

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        print("Local model dir not found: {}. Train and save the model first.".format(local_dir))
        sys.exit(1)

    from huggingface_hub import HfApi
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=args.private, exist_ok=True)

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
    )

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(local_dir))
        tokenizer.push_to_hub(repo_id)
    except Exception:
        pass

    print("Pushed to https://huggingface.co/{}".format(repo_id))
    print("Ensure the model is publicly accessible (Settings -> Visibility) if required.")


if __name__ == "__main__":
    main()
