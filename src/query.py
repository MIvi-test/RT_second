import json
from pathlib import Path
from search import semantic_search, hybrid_search

SCRIPT_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = SCRIPT_DIR / "dataset_case3_v1.0_fix" / "eval_questions.json"
RESULTS_PATH = SCRIPT_DIR / "results.json"

# Choose search method for final results: 'semantic' or 'hybrid'
SEARCH_METHOD = "semantic"

questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
print(f"Method: {SEARCH_METHOD} | Questions: {len(questions)}\n")

results = []
for q in questions:
    # get top hits using chosen search method
    if SEARCH_METHOD == "hybrid":
        hits = hybrid_search(q["query"], top_k=5)
    else:
        hits = semantic_search(q["query"], top_k=5)

    top5 = [h["chunk_id"] for h in hits]

    results.append({
        "question_id": q["question_id"],
        "top_5_chunks": top5,
    })

    print(f"[{q['question_id']}] {q['query'][:65]}")
    for h in hits:
        print(f"   {h.get('score', 0):5.1f}%  {h['name']}  ({h['file_path']})")
    print()

RESULTS_PATH.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"Done → {RESULTS_PATH}")


__all__ = [
    "SEARCH_METHOD",
    "questions",
    "results",
]