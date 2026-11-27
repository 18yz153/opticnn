"""
Prepare Toy Math Addition Dataset
=================================
Generates addition equations: "A+B=C\\n" (e.g. "15+32=047\\n")
Vocabulary has 13 tokens: '0'-'9', '+', '=', '\\n'
Tests attention data routing:
The model must attend to digits of A and B to predict the sum.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import random

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DATA_DIR, "toy_math.txt")

def generate_math_dataset(num_samples=10000, seed=42):
    random.seed(seed)
    lines = []
    for _ in range(num_samples):
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        c = a + b
        line = f"{a:02d}+{b:02d}={c:03d}\n"
        lines.append(line)

    with open(FILE_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Toy math dataset generated: {FILE_PATH} ({len(lines)} samples)")

if __name__ == "__main__":
    generate_math_dataset()
