"""AbstentionResponder - terminal component for the abstain branch.

Consumes the router's `insufficient` payload and turns it into a clear,
structured reply that names the gap instead of guessing. Wire it directly:

    router.insufficient -> responder.insufficient
"""
from typing import Any, Dict, Optional

from haystack import component


@component
class AbstentionResponder:
    def __init__(self, template: Optional[str] = None):
        # {query} and {reason} are filled from the router payload.
        self.template = template or (
            "I don't have enough information in the knowledge base to answer "
            "\"{query}\" reliably — {reason}. I'm flagging this rather than guessing."
        )

    @component.output_types(reply=str, meta=Dict[str, Any])
    def run(self, insufficient: Dict[str, Any]) -> Dict[str, Any]:
        query = insufficient.get("query", "")
        reason = insufficient.get("reason", "the retrieved context was insufficient")
        reply = self.template.format(query=query, reason=reason)
        return {
            "reply": reply,
            "meta": {
                "abstained": True,
                "verdict": insufficient.get("verdict", "GAP"),
                "signals": insufficient.get("signals", {}),
            },
        }
