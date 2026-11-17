"""
准备玩具级算术数据集 (Toy Math Addition Benchmark)
===================================================
任务设计:
  生成两位数加法题: "A+B=C\\n" (例如 "15+32=047\\n")
  总词表只有 13 个字符: '0'-'9', '+', '=', '\\n'
  测试核心: 注意力机制的“精准数据路由 (Data Routing)”能力。
  模型必须将注意力精准地聚焦在 A 的各位和 B 的各位数字上，并完成进位加法。
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
    # 穷举或随机采样两位数加法
    seen = set()
    for _ in range(num_samples):
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        c = a + b
        # 格式化: 两位数加法, 和用三位数零填充 (例如 "15+32=047\n")
        line = f"{a:02d}+{b:02d}={c:03d}\n"
        lines.append(line)

    with open(FILE_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"✅ 玩具级加法数据集生成完成: {FILE_PATH} (共 {len(lines)} 条算术样本)")

if __name__ == "__main__":
    generate_math_dataset()
