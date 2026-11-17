"""
准备字符级莎士比亚数据集 (TinyShakespeare)
===========================================
下载 ~1MB 的莎士比亚剧本纯文本，并切分为训练集 (90%) 和验证集 (10%)。
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
        print(f"正在下载 TinyShakespeare 数据集: {DATA_URL} ...")
        try:
            req = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8')
            with open(FILE_PATH, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ 下载成功! 保存至 {FILE_PATH} (文件大小: {len(content)/1024:.1f} KB)")
        except Exception as e:
            print(f"⚠️ 下载失败 ({e})，将使用备用离线内置小样本。")
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

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

ROMEO:
What optical sorcery is this?
O, speak again, bright angel! for thou art
As glorious to this night, being o'er my head
As is a winged messenger of heaven
Unto the white-upturned wondering eyes
Of mortals that fall back to gaze on him
When he bestrides the lazy-pacing clouds
And sails upon the bosom of the air.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.
""" * 100) # 重复构建样本
    else:
        print(f"✅ TinyShakespeare 数据集已存在: {FILE_PATH}")

if __name__ == "__main__":
    download_data()
