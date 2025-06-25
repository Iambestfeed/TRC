from huggingface_hub import login
import sys

from huggingface_hub import snapshot_download
repo_name = sys.argv[1]
print(repo_name)
snapshot_download(repo_id=repo_name,
                  local_dir="./data",
                  local_dir_use_symlinks=False,
                  repo_type='dataset')
