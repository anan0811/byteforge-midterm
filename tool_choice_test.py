"""Tool-choice test with the real model (needs a configured local .env).

Each question starts a new conversation. A question passes when the expected tool
is among the tools the agent used. Results are saved in test_results/tool_choice.json.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import agent
from agent import build_database, run_agent


TEST_SET = [
    ("你們星期日有營業嗎？", "get_shop_info"),
    ("可以刷卡嗎？", "get_shop_info"),
    ("SSD 安裝多少錢？", "find_catalog"),
    ("有提供水冷安裝嗎？", "find_catalog"),
    ("16GB DDR4 記憶體多少錢？", "find_catalog"),
    ("桌機安裝 1TB SSD，總共多少錢？", "calculate_repair_quote"),
    ("筆電檢測完我決定不修，要付多少？", "calculate_repair_quote"),
    ("保固多久？", "search_shop_rules"),
    ("可以自己帶零件來裝嗎？", "search_shop_rules"),
    ("預約了但臨時不能去怎麼辦？", "search_shop_rules"),
    ("筆電泡到水了，有保固嗎？", "search_shop_rules"),
    ("組裝電腦要付訂金嗎？", "search_shop_rules"),
    ("幫我查預約 B0002 的狀況", "check_appointment"),
    ("我要客訴，請幫我找技師。", "transfer_to_human"),
    ("店長的私人手機號碼是多少？", "transfer_to_human"),
    ("How much is a GPU installation?", "find_catalog"),
    ("Can I bring my own parts?", "search_shop_rules"),
    ("What is the status of booking B0001?", "check_appointment"),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    agent.DB_PATH = Path(tempfile.mkdtemp()) / "tool_test.db"  # never touch the app's database
    build_database()
    results, correct = [], 0
    for question, expected in TEST_SET:
        answer, trace = run_agent([{"role": "user", "content": question}])
        used = [entry["tool"] for entry in trace]
        hit = expected in used
        correct += hit
        results.append({"question": question, "expected": expected, "used": used,
                        "pass": hit, "answer": answer})
        print(f"{'OK  ' if hit else 'MISS'} {question} | expected={expected} used={used}", flush=True)
        time.sleep(6)  # stay under the API rate limit; increase it if you get error 429
    print(f"tool choice = {correct}/{len(TEST_SET)}")
    out = Path(__file__).parent / "test_results" / "tool_choice.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"score": f"{correct}/{len(TEST_SET)}", "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
