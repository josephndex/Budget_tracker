# models.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

@dataclass
class Transaction:
    timestamp: datetime
    amount: float
    category: str
    note: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "amount": float(self.amount),
            "category": self.category,
            "note": self.note or ""
        }

    @staticmethod
    def from_dict(d):
        ts = d["timestamp"]
        if isinstance(ts, datetime):
            timestamp = ts
        else:
            timestamp = datetime.fromisoformat(str(ts))
        return Transaction(
            id=d.get("id", str(uuid.uuid4())),
            timestamp=timestamp,
            amount=float(d["amount"]),
            category=d.get("category",""),
            note=d.get("note","")
        )


# Category management (for color, budget, etc.)
@dataclass
class Category:
    name: str
    color: str = "#1976d2"  # Default blue
    budget: float = 0.0

    def to_dict(self):
        return {"name": self.name, "color": self.color, "budget": self.budget}

    @staticmethod
    def from_dict(d):
        return Category(
            name=d["name"],
            color=d.get("color", "#1976d2"),
            budget=float(d.get("budget", 0.0))
        )
