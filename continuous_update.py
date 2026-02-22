import os
import subprocess
from deltalake import DeltaTable

LAST_VERSION_FILE = "last_trained_gold_version.txt"

def get_latest_gold_version():
    """Returns the highest version number of the Gold table."""
    dt = DeltaTable("data/delta/gold")
    history = dt.history()
    versions = [entry['version'] for entry in history]
    return max(versions)

def get_last_trained_version():
    """Reads the last trained version from file, or returns -1 if not exists."""
    if os.path.exists(LAST_VERSION_FILE):
        with open(LAST_VERSION_FILE, 'r') as f:
            return int(f.read().strip())
    return -1

def train_on_version(version):
    """Trains a model on the given Gold version using train_updated.py."""
    print(f"New Gold version {version} detected. Training new model...")
    # You can either call train_updated.py (which uses a fixed version) or modify it to accept a version argument.
    # For flexibility, we'll create a separate script that takes version as argument.
    # We'll assume you have a script 'train_on_version.py' that accepts --version.
    subprocess.run(["python", "train_on_version.py", "--version", str(version)])

def main():
    latest = get_latest_gold_version()
    last = get_last_trained_version()
    if latest > last:
        train_on_version(latest)
        with open(LAST_VERSION_FILE, 'w') as f:
            f.write(str(latest))
    else:
        print(f"No new Gold version. Latest: {latest}, last trained: {last}")

if __name__ == "__main__":
    main()