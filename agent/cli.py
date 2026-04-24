from __future__ import annotations

import argparse
import logging

from agent.graph import AgentConfig, MultiMemoryAgent
from config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-memory LangGraph agent CLI")
    parser.add_argument("--user-id", default="local", help="User id for memory namespaces")
    parser.add_argument("--no-memory", action="store_true", help="Disable all memories (baseline)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = get_settings()
    agent = MultiMemoryAgent(settings=settings, config=AgentConfig(enable_memory=not args.no_memory))

    print("Multi-memory agent (LangGraph). Commands: /reset, /clear-all, /exit")
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            agent.reset(args.user_id)
            print("Assistant: (short-term cleared)")
            continue
        if user == "/clear-all":
            agent.clear_all(args.user_id)
            print("Assistant: (all memories cleared)")
            continue

        state = agent.run(user_id=args.user_id, user_input=user)
        print("Assistant:", state.get("response", ""))


if __name__ == "__main__":
    main()

