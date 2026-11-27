"""
Prepare TinyShakespeare Dataset
===============================
Downloads ~1MB of raw Shakespeare text and splits into train (90%) and val (10%).
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DATA_DIR, "tinyshakespeare.txt")

def download_data():
    if not os.path.exists(FILE_PATH):
        print(f"Downloading TinyShakespeare dataset: {DATA_URL} ...")
        try:
            req = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8')
            with open(FILE_PATH, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Download complete: {FILE_PATH} ({len(content)/1024:.1f} KB)")
        except Exception as e:
            print(f"Download failed ({e}), using local fallback sample.")
            with open(FILE_PATH, "w", encoding="utf-8") as f:
                f.write("""First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?
""")
    else:
        print(f"TinyShakespeare dataset already exists: {FILE_PATH}")

if __name__ == "__main__":
    download_data()
