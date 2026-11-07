"""Print the real embedding scores of rule questions and off-topic questions (used to choose MIN_SCORE)."""
import sys
import numpy as np
import agent

sys.stdout.reconfigure(encoding="utf-8")
chunks, vectors = agent.rule_index()
print("index:", vectors.shape)
questions = [
    # (question, expected chunk title or None)
    ("保固多久？", "保固"),
    ("我可以自己帶零件來裝嗎？", "自備零件"),
    ("預約了但臨時不能去怎麼辦？", "預約的取消與改期"),
    ("檢測完我不想修，要付錢嗎？", "檢測費與拒絕維修"),
    ("筆電泡到水了可以修嗎？", "進液與摔落等實體損壞"),
    ("修好了我很久沒去拿會怎樣？", "取件與逾期未取件"),
    ("你們會看我電腦裡的照片嗎？", "個人資料與隱私"),
    ("組裝電腦要付訂金嗎？", "電腦組裝訂單"),
    ("換下來的舊硬碟會還我嗎？", "更換下來的舊零件"),
    ("修電腦大概要幾天？", "維修與檢測所需時間"),
    ("報價可以保留多久？", "報價與同意維修"),
    ("資料會不會不見？", "資料備份與資料轉移"),
    ("How do I book a repair?", "預約送修"),
    ("Is water damage covered by warranty?", "進液與摔落等實體損壞"),
    ("明天高雄天氣如何？", None),
    ("推薦我一部好看的電影", None),
    ("台積電股價多少？", None),
    ("Who is the CEO of NVIDIA?", None),
    ("你們有賣手機殼嗎？", None),
]
titles = [c["title"] for c in chunks]
for q, expected in questions:
    v = agent.embed_texts([q])[0]
    s = vectors @ v / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(v))
    order = np.argsort(s)[::-1]
    top = [(titles[i], round(float(s[i]), 3)) for i in order[:2]]
    ok = "" if expected is None else ("OK " if titles[order[0]] == expected else "MISS")
    print(f"{ok:4} {q:<28} {top}")
