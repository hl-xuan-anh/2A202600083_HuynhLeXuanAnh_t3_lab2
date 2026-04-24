from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from agent.graph import AgentConfig, MultiMemoryAgent
from benchmark.evaluator import compute_turn_metrics
from benchmark.test_conversations import CASES
from config import get_settings


log = logging.getLogger(__name__)


def run_case(agent: MultiMemoryAgent, user_id: str, turns: List[str]) -> Dict[str, Any]:
    last_state: Dict[str, Any] = {}
    for t in turns:
        # Use plain text commands for the CLI; benchmark drives the agent directly.
        last_state = agent.run(user_id=user_id, user_input=t)
    return last_state


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()

    out_dir = Path("benchmark") / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{ts}"

    with_mem = MultiMemoryAgent(settings=settings, config=AgentConfig(enable_memory=True))
    no_mem = MultiMemoryAgent(settings=settings, config=AgentConfig(enable_memory=False))

    # Use isolated user_ids so runs don't contaminate each other.
    with_user = f"bench_with_{run_id}"
    no_user = f"bench_no_{run_id}"
    with_mem.clear_all(with_user)
    no_mem.clear_all(no_user)

    rows: List[Dict[str, Any]] = []
    for case in CASES:
        with_mem.reset(with_user)
        no_mem.reset(no_user)

        state_no = run_case(no_mem, no_user, case.turns)
        state_with = run_case(with_mem, with_user, case.turns)

        m_no = compute_turn_metrics(state_no, case.expected_substring)
        m_with = compute_turn_metrics(state_with, case.expected_substring)

        rows.append(
            {
                "case_id": case.case_id,
                "scenario": case.scenario,
                "expected": case.expected_substring or "",
                "no_memory_response": state_no.get("response", ""),
                "with_memory_response": state_with.get("response", ""),
                "no_memory_relevance": m_no.relevance_score,
                "with_memory_relevance": m_with.relevance_score,
                "no_memory_used_memory": m_no.used_memory,
                "with_memory_used_memory": m_with.used_memory,
                "no_memory_tokens": m_no.token_total,
                "with_memory_tokens": m_with.token_total,
                "with_memory_hits": dict(m_with.memory_hits),
                "intent": state_with.get("intent", ""),
            }
        )

    # Aggregate summary
    avg = lambda k: sum(float(r[k]) for r in rows) / max(1, len(rows))
    summary = {
        "run_id": run_id,
        "cases": len(rows),
        "avg_no_memory_relevance": avg("no_memory_relevance"),
        "avg_with_memory_relevance": avg("with_memory_relevance"),
        "avg_no_memory_tokens": avg("no_memory_tokens"),
        "avg_with_memory_tokens": avg("with_memory_tokens"),
        "with_memory_pass_rate": sum(1 for r in rows if r["with_memory_relevance"] >= 1.0) / max(1, len(rows)),
    }

    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"
    md_path = out_dir / "summary.md"

    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case_id",
            "scenario",
            "expected",
            "no_memory_response",
            "with_memory_response",
            "no_memory_relevance",
            "with_memory_relevance",
            "no_memory_used_memory",
            "with_memory_used_memory",
            "no_memory_tokens",
            "with_memory_tokens",
            "intent",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    md_lines: List[str] = []
    md_lines.append("# Benchmark Summary\n")
    md_lines.append(f"- Run id: `{run_id}`")
    md_lines.append(f"- Cases: {summary['cases']}")
    md_lines.append(f"- Avg relevance (no-memory): {summary['avg_no_memory_relevance']:.2f}")
    md_lines.append(f"- Avg relevance (with-memory): {summary['avg_with_memory_relevance']:.2f}")
    md_lines.append(f"- With-memory pass rate: {summary['with_memory_pass_rate']:.2%}")
    md_lines.append(f"- Avg tokens (no-memory): {summary['avg_no_memory_tokens']:.0f}")
    md_lines.append(f"- Avg tokens (with-memory): {summary['avg_with_memory_tokens']:.0f}\n")
    md_lines.append("## Per-case (with-memory)\n")
    for r in rows:
        md_lines.append(f"- #{r['case_id']} {r['scenario']} | intent={r['intent']} | hits={r['with_memory_hits']}")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

