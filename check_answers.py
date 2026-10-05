import json
import sys
import urllib.request

import agent
from agent import YourAgent

STYLE = getattr(agent, "ANSWER_STYLE", "")[:40]

URL = "https://app.geckovision.tech/api/dev3pack/questions?set=final"
REFUSAL = ("don't know", "do not know")

questions = json.load(urllib.request.urlopen(URL))["questions"]
wanted = sys.argv[1:] or ["pf-10", "pf-12", "pf-15"]
if wanted == ["all"]:
    wanted = [q["task_id"] for q in questions]
agent = YourAgent()
ok_count = 0
for q in (q for q in questions if q["task_id"] in wanted):
    answer = agent(q["question"])
    refused = any(sign in answer.answer.lower() for sign in REFUSAL)
    if STYLE and STYLE in answer.answer:
        verdict = "BUG: the answer is your own instructions"
    elif q["expected_behavior"] == "refuse":
        verdict = "ok: refused" if refused else "WRONG: should refuse"
    elif refused:
        verdict = "WRONG: should answer, it refused"
    elif not answer.citations:
        verdict = "WRONG: answered without citations"
    elif answer.needs_human_review:
        verdict = "WRONG: answered but flagged for review"
    else:
        verdict = "ok: answered with " + ", ".join(answer.citations)
    ok_count += verdict.startswith("ok")
    print(f"{q['task_id']}  ({q['category']}, expects: {q['expected_behavior']})  ->  {verdict}")
    print(f"      {repr(answer.answer[:120])}")
print(f"\n{ok_count}/{len(wanted)} as expected")
