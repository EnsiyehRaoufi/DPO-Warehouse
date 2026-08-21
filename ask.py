"""Live demo of the routing and answering logic, with no pre-known
intent - a raw question goes in, a routed and grounded answer comes
out. No model is involved; this exercises the rule-based router and
tool functions directly.

Usage:
    python ask.py "any active stockouts?"
    python ask.py "why is WH_0009 at the national_CMS failing fulfillment for vaccines?"
    python ask.py "how is WH_0009 doing overall?"
"""

import sys

import assistant as A
import data_utils as du


def main():
    if len(sys.argv) < 2:
        print("usage: python ask.py \"your question here\"")
        recs = du.load_data(use_sample=True)
        print("\nexample warehouse ids in the sample data (same id can exist at multiple levels!):")
        for uid in sorted({r["uid"] for r in recs})[:5]:
            print(" ", uid)
        return

    question = sys.argv[1]
    recs = du.load_data(use_sample=True)

    intent, scope = A.route(question, recs)
    print(f"routed to: {intent}, scope={scope}\n")

    answer, tool_out = A.answer_from_text(recs, question)
    print(answer)


if __name__ == "__main__":
    main()
