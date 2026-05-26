"""
Unit tests for src/linked_view/router.py

Covers pure-logic functions that have no I/O dependencies:
  - BinaryRouter._extract_features
  - BinaryRouter._heuristic_p_raw
  - LLMRouter.parse_thinking_output  (static method, same impl in ThinkingLLMRouter)
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.linked_view.summary_index import SummaryHit, EvidenceSet
from src.linked_view.router import BinaryRouter, LLMRouter, ThinkingLLMRouter


def _make_hit(summary_text="some text", score=0.5, raw_log_id="id1",
              timestamp=None, session_id=None):
    return SummaryHit(
        summary_text=summary_text,
        score=score,
        raw_log_id=raw_log_id,
        timestamp=timestamp,
        session_id=session_id,
        extra={},
    )


def _evidence(query, hits):
    return EvidenceSet(query=query, hits=hits)


# ---------------------------------------------------------------------------
# _extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def setup_method(self):
        self.router = BinaryRouter()

    def test_returns_nine_features(self):
        ev = _evidence("test query", [_make_hit(score=0.8)])
        feats = self.router._extract_features(ev)
        assert len(feats) == 9

    def test_empty_hits_zero_scores(self):
        ev = _evidence("q", [])
        feats = self.router._extract_features(ev)
        s_max, s_mean, s_std, entropy = feats[:4]
        assert s_max == 0.0
        assert s_mean == 0.0
        assert s_std == 0.0
        assert entropy == 0.0

    def test_single_hit_values(self):
        hit = _make_hit(score=0.6, summary_text="hello world", session_id="s1")
        ev = _evidence("one two three", [hit])
        feats = self.router._extract_features(ev)
        s_max, s_mean, s_std, entropy, total_sum_len, hit_ratio, has_ts, multi_session, q_len = feats
        assert s_max == 0.6
        assert s_mean == 0.6
        assert s_std == 0.0
        assert total_sum_len == len("hello world")
        assert hit_ratio == 1
        assert has_ts == 0.0       # no timestamp
        assert multi_session == 0.0
        assert q_len == 3

    def test_multi_session_flag(self):
        hits = [
            _make_hit(score=0.5, session_id="s1"),
            _make_hit(score=0.5, session_id="s2"),
        ]
        ev = _evidence("q", hits)
        feats = self.router._extract_features(ev)
        assert feats[7] == 1.0    # multi_session

    def test_single_session_flag(self):
        hits = [
            _make_hit(score=0.5, session_id="s1"),
            _make_hit(score=0.5, session_id="s1"),
        ]
        ev = _evidence("q", hits)
        feats = self.router._extract_features(ev)
        assert feats[7] == 0.0    # NOT multi_session

    def test_has_timestamp_flag(self):
        hit = _make_hit(score=0.5, timestamp="2024-01-01 09:00:00")
        ev = _evidence("q", [hit])
        feats = self.router._extract_features(ev)
        assert feats[6] == 1.0    # has_ts

    def test_entropy_uniform_distribution(self):
        hits = [_make_hit(score=1.0) for _ in range(4)]
        ev = _evidence("q", hits)
        feats = self.router._extract_features(ev)
        entropy = feats[3]
        expected = -4 * (0.25 * math.log(0.25 + 1e-8))
        assert abs(entropy - expected) < 1e-6

    def test_query_length_tokenised_by_whitespace(self):
        ev = _evidence("what is the capital of France", [_make_hit()])
        feats = self.router._extract_features(ev)
        assert feats[8] == 6


# ---------------------------------------------------------------------------
# _heuristic_p_raw
# ---------------------------------------------------------------------------

class TestHeuristicPRaw:
    def setup_method(self):
        self.router = BinaryRouter()

    def _run(self, query, hits=None):
        if hits is None:
            hits = [_make_hit(score=0.5) for _ in range(5)]
        ev = _evidence(query, hits)
        feats = self.router._extract_features(ev)
        return self.router._heuristic_p_raw(ev, feats)

    def test_result_in_unit_interval(self):
        score = self._run("simple question")
        assert 0.0 <= score <= 1.0

    def test_long_query_boosts_score(self):
        short_score = self._run("short q")
        long_score = self._run(" ".join(["word"] * 25))
        assert long_score > short_score

    def test_temporal_keyword_boosts_score(self):
        base = self._run("what did they say")
        with_temporal = self._run("最近发生了什么")
        assert with_temporal > base

    def test_causal_keyword_boosts_score(self):
        base = self._run("what happened")
        with_causal = self._run("why did this happen and how did it start")
        assert with_causal > base

    def test_lack_evidence_boosts_score(self):
        good_hits = [_make_hit(score=0.9) for _ in range(5)]
        poor_hits = [_make_hit(score=0.1)]          # only 1 hit, low score
        q = "simple question"
        good_ev = _evidence(q, good_hits)
        good_feats = self.router._extract_features(good_ev)
        poor_ev = _evidence(q, poor_hits)
        poor_feats = self.router._extract_features(poor_ev)
        score_good = self.router._heuristic_p_raw(good_ev, good_feats)
        score_poor = self.router._heuristic_p_raw(poor_ev, poor_feats)
        assert score_poor > score_good

    # --- medical keyword additions (added in router.py) ---

    def test_lab_value_keyword_boosts_score(self):
        base = self._run("how is the patient doing")
        lab = self._run("what is the patient creatinine level")
        assert lab > base

    def test_completeness_keyword_boosts_score(self):
        base = self._run("what symptoms were reported")
        complete = self._run("list all symptoms the patient mentioned")
        assert complete > base

    def test_cross_visit_keyword_boosts_score(self):
        base = self._run("what symptoms were reported")
        cross = self._run("what did the patient previously mentioned about their pain")
        assert cross > base

    def test_threshold_decision(self):
        router_low = BinaryRouter(threshold=0.0)
        ev = _evidence("simple q", [_make_hit(score=0.9) for _ in range(5)])
        assert router_low.decide(ev) == "R"   # p_raw > 0.0 always

        router_high = BinaryRouter(threshold=1.0)
        assert router_high.decide(ev) == "S"  # p_raw never > 1.0


# ---------------------------------------------------------------------------
# parse_thinking_output  (static, identical in LLMRouter and ThinkingLLMRouter)
# ---------------------------------------------------------------------------

class TestParseThinkingOutput:
    @staticmethod
    def _parse(text):
        return LLMRouter.parse_thinking_output(text)

    def test_extracts_content_after_close_tag(self):
        text = "<think>some reasoning\nmore reasoning</think>S"
        assert self._parse(text) == "S"

    def test_strips_whitespace_after_tag(self):
        text = "<think>reasoning</think>   \n  R  \n"
        assert self._parse(text) == "R"

    def test_no_think_tag_returns_original(self):
        text = "  just a plain answer  "
        assert self._parse(text) == "just a plain answer"

    def test_empty_string_returns_empty(self):
        assert self._parse("") == ""

    def test_none_equivalent_empty(self):
        # parse_thinking_output checks `if not text`
        assert self._parse("") == ""

    def test_unclosed_think_tag_returns_empty(self):
        text = "<think>still thinking..."
        assert self._parse(text) == ""

    def test_multiline_content_after_tag(self):
        text = "<think>reasoning</think>\n{\"action\": \"R\"}"
        result = self._parse(text)
        assert result == '{"action": "R"}'

    def test_case_insensitive_tag(self):
        text = "<THINK>reasoning</THINK>answer"
        assert self._parse(text) == "answer"

    def test_thinking_llm_router_same_impl(self):
        text = "<think>x</think>S"
        assert ThinkingLLMRouter.parse_thinking_output(text) == "S"
