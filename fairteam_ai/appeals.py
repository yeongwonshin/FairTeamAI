from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal

AppealStatus = Literal["submitted", "under_review", "accepted", "rejected"]


@dataclass
class Appeal:
    appeal_id: str
    member: str
    category: str
    claim: str
    evidence_ref: str
    status: AppealStatus = "submitted"
    reviewer_note: str = ""
    created_at: str = ""
    updated_at: str = ""


class AppealStore:
    """Tiny JSONL-backed appeal workflow for demos and classroom pilots."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> List[Appeal]:
        if not self.path.exists():
            return []
        appeals: List[Appeal] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                appeals.append(Appeal(**json.loads(line)))
            except Exception:
                continue
        return appeals

    def submit(self, member: str, category: str, claim: str, evidence_ref: str) -> Appeal:
        now = datetime.now(timezone.utc).isoformat()
        appeal = Appeal(
            appeal_id=str(uuid.uuid4())[:8],
            member=member.strip(),
            category=category.strip(),
            claim=claim.strip(),
            evidence_ref=evidence_ref.strip(),
            created_at=now,
            updated_at=now,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(appeal), ensure_ascii=False) + "\n")
        return appeal

    def update(self, appeal_id: str, status: AppealStatus, reviewer_note: str = "") -> Appeal | None:
        appeals = self.list()
        updated: Appeal | None = None
        for appeal in appeals:
            if appeal.appeal_id == appeal_id:
                appeal.status = status
                appeal.reviewer_note = reviewer_note
                appeal.updated_at = datetime.now(timezone.utc).isoformat()
                updated = appeal
        self.path.write_text("\n".join(json.dumps(asdict(a), ensure_ascii=False) for a in appeals) + ("\n" if appeals else ""), encoding="utf-8")
        return updated

    def summary_by_member(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for appeal in self.list():
            counts[appeal.member] = counts.get(appeal.member, 0) + 1
        return counts
