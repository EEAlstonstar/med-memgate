"""
Unit tests for src/linked_view/pipelines_slow.py

Tests the pure logic in build_evidence:
  - skips hits with no raw_log_id
  - skips hits whose raw_log_id has no matching record in RawStore
  - correctly assembles EvidenceDoc fields from hit + raw record
"""

import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.linked_view.summary_index import SummaryHit
from src.linked_view.pipelines_slow import build_evidence, EvidenceDoc


def _make_hit(raw_log_id, summary_text="summary", score=0.7, timestamp=None):
    return SummaryHit(
        summary_text=summary_text,
        score=score,
        raw_log_id=raw_log_id,
        timestamp=timestamp,
        session_id=None,
        extra={},
    )


def _make_raw_record(raw_log_id, text):
    rec = MagicMock()
    rec.raw_log_id = raw_log_id
    rec.text = text
    return rec


def _make_raw_store(records):
    """Return a mock RawStore whose batch_get returns `records`."""
    store = MagicMock()
    store.batch_get.return_value = records
    return store


# ---------------------------------------------------------------------------
# build_evidence
# ---------------------------------------------------------------------------

class TestBuildEvidence:
    def test_basic_assembly(self):
        hit = _make_hit("r1", summary_text="the summary", score=0.9, timestamp="2024-01-01")
        raw = _make_raw_record("r1", "raw conversation text")
        store = _make_raw_store([raw])

        docs = build_evidence(store, [hit])

        assert len(docs) == 1
        doc = docs[0]
        assert isinstance(doc, EvidenceDoc)
        assert doc.raw_log_id == "r1"
        assert doc.summary == "the summary"
        assert doc.raw_text == "raw conversation text"
        assert doc.score == 0.9
        assert doc.timestamp == "2024-01-01"

    def test_skips_hit_with_no_raw_log_id(self):
        hit_ok = _make_hit("r1")
        hit_none = _make_hit(None)
        raw = _make_raw_record("r1", "text")
        store = _make_raw_store([raw])

        docs = build_evidence(store, [hit_ok, hit_none])

        assert len(docs) == 1
        assert docs[0].raw_log_id == "r1"

    def test_skips_hit_when_raw_record_missing(self):
        hit1 = _make_hit("r1")
        hit2 = _make_hit("r2")        # r2 has NO matching raw record
        raw1 = _make_raw_record("r1", "text for r1")
        store = _make_raw_store([raw1])  # only r1 returned

        docs = build_evidence(store, [hit1, hit2])

        assert len(docs) == 1
        assert docs[0].raw_log_id == "r1"

    def test_empty_hits_returns_empty(self):
        store = _make_raw_store([])
        docs = build_evidence(store, [])
        assert docs == []

    def test_all_hits_missing_raw(self):
        hits = [_make_hit("r1"), _make_hit("r2")]
        store = _make_raw_store([])   # nothing found in store

        docs = build_evidence(store, hits)
        assert docs == []

    def test_order_preserved(self):
        hits = [_make_hit("r1"), _make_hit("r2"), _make_hit("r3")]
        raws = [
            _make_raw_record("r1", "text1"),
            _make_raw_record("r2", "text2"),
            _make_raw_record("r3", "text3"),
        ]
        store = _make_raw_store(raws)

        docs = build_evidence(store, hits)

        assert len(docs) == 3
        assert [d.raw_log_id for d in docs] == ["r1", "r2", "r3"]

    def test_batch_get_called_with_correct_ids(self):
        hits = [_make_hit("r1"), _make_hit("r2"), _make_hit(None)]
        store = _make_raw_store([])

        build_evidence(store, hits)

        # None-id hit must not be passed to batch_get
        store.batch_get.assert_called_once_with(["r1", "r2"])

    def test_multiple_hits_same_raw_id(self):
        hit1 = _make_hit("r1", summary_text="summary A", score=0.8)
        hit2 = _make_hit("r1", summary_text="summary B", score=0.5)
        raw = _make_raw_record("r1", "shared raw text")
        store = _make_raw_store([raw])

        docs = build_evidence(store, [hit1, hit2])

        # Both hits resolve to the same raw record — both should be included
        assert len(docs) == 2
        assert docs[0].summary == "summary A"
        assert docs[1].summary == "summary B"
        assert docs[0].raw_text == docs[1].raw_text == "shared raw text"
