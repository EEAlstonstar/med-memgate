#!/usr/bin/env python3
"""
Convert biomedical QA pairs (data.json) to med-memgate multi-turn session format (sessions.jsonl).

Each QA pair becomes one session with:
  - Multiple "visits" (turns) distributing answer entities across timestamps
  - Up to 3 QA types: complete (cat=0, route=R), first_visit (cat=1, route=R), latest (cat=2, route=S)

Usage:
    python scripts/biomedical/1_convert_to_sessions.py \
        --input core/datasets/biomedical/data.json \
        --output core/datasets/biomedical/sessions.jsonl
"""

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional


# ── Question type classification ──────────────────────────────────────────────

def get_qtype(qid: int) -> str:
    if qid <= 9:  return "side_effect"
    if qid <= 19: return "disease_symptom"
    if qid <= 29: return "gene_bio_process"
    if qid <= 39: return "gene_mol_function"
    if qid <= 49: return "gene_anatomy_down"
    if qid <= 59: return "gene_anatomy_expr"
    if qid <= 69: return "gene_anatomy_up"
    if qid <= 79: return "drug_compound"   # "which compound treats both X and Y" → compound is answer
    return "drug_anatomy"                   # "what disease in anatomy can compound X treat" → disease is answer


# ── Subject / entity extraction ───────────────────────────────────────────────

def _fmt(entities: List[str]) -> str:
    if len(entities) == 1:
        return entities[0]
    if len(entities) == 2:
        return f"{entities[0]} and {entities[1]}"
    return ", ".join(entities[:-1]) + f", and {entities[-1]}"


def extract_subject_entities(question: str, answer: str, qtype: str) -> Tuple[str, List[str]]:
    """Return (subject, entities) where entities are the items to spread across visits."""
    q = question

    if qtype == "side_effect":
        m = (re.search(r'(?:using\s+compound\s+|compound\s+|the\s+compound\s+)(.+?)(?:\?|$)', q, re.I)
             or re.search(r'side effects? of\s+([A-Z][A-Za-z\s]+?)(?:\?|$)', q))
        subject = m.group(1).strip() if m else answer.strip()
        entities = [e.strip() for e in answer.split(",") if e.strip()]

    elif qtype == "disease_symptom":
        m = (re.search(r'disease\s+([\w\s]+?)(?:\?|$)', q, re.I)
             or re.search(r'if I have\s+([\w\s]+?)(?:\?|$)', q, re.I)
             or re.search(r'symptoms? of\s+([\w\s]+?)(?:\?|\.|are\b|$)', q, re.I))
        subject = m.group(1).strip() if m else "the condition"
        entities = [e.strip() for e in answer.split(",") if e.strip()]

    elif qtype in ("gene_bio_process", "gene_mol_function",
                   "gene_anatomy_down", "gene_anatomy_expr", "gene_anatomy_up"):
        m = (re.search(r'gene\s+([A-Z][A-Z0-9]+\w*)', q)
             or re.search(r'the\s+([A-Z][A-Z0-9]+\w*)\s+gene', q))
        subject = m.group(1) if m else "the gene"
        entities = [e.strip() for e in answer.split(",") if e.strip()]

    elif qtype == "drug_compound":
        # Answer is the compound; diseases come from question ("both X and Y")
        subject = answer.strip()
        m = re.search(r'both\s+(.+?)\s+and\s+(.+?)(?:\?|$)', q, re.I)
        if m:
            entities = [m.group(1).strip(), m.group(2).strip()]
        else:
            entities = [e.strip() for e in answer.split(",") if e.strip()]

    else:  # drug_anatomy
        # Answer is the disease; compound comes from question
        for pat in [r'(?:by|with)\s+([A-Z][A-Za-z\s\-]+?)(?:\?|$)',
                    r'use of\s+([A-Z][A-Za-z\s\-]+?)\s+relieve',
                    r'can\s+([A-Z][A-Za-z\s\-]+?)\s+(?:help|treat)']:
            m = re.search(pat, q)
            if m:
                break
        subject = m.group(1).strip() if m else "the compound"
        entities = [answer.strip()]

    return subject, entities or [answer.strip()]


# ── Dialogue turn templates ────────────────────────────────────────────────────

_TURN_TEMPLATES: Dict[str, Tuple[str, str]] = {
    "side_effect": (
        "Doctor, I've been taking {subject} as prescribed. I've noticed {entities}.",
        "Thank you for letting me know. I'll document {entities} as side effects of {subject}.",
    ),
    "disease_symptom": (
        "Doctor, I've been experiencing {entities} lately.",
        "I see. {entities_cap} can be associated with {subject}. I've noted these in your chart.",
    ),
    "gene_bio_process": (
        "I'm here for my follow-up on the {subject} gene panel.",
        "Your report shows the {subject} gene is involved in: {entities}.",
    ),
    "gene_mol_function": (
        "Can you explain the molecular function results for the {subject} gene?",
        "The analysis reveals the {subject} gene plays a role in: {entities}.",
    ),
    "gene_anatomy_down": (
        "I wanted to review the latest findings on the {subject} gene.",
        "The study shows the {subject} gene can downregulate: {entities}.",
    ),
    "gene_anatomy_expr": (
        "What does the expression report say about the {subject} gene?",
        "The report shows the {subject} gene is expressed in: {entities}.",
    ),
    "gene_anatomy_up": (
        "Can you walk me through the upregulation results for {subject}?",
        "The test shows the {subject} gene upregulates: {entities}.",
    ),
    "drug_compound": (
        "I wanted to discuss conditions treatable with {subject}.",
        "{subject} has shown effectiveness against {entities}. We can include this in your treatment plan.",
    ),
    "drug_anatomy": (
        "I've been having health concerns and my doctor mentioned {subject}.",
        "Based on your case, {subject} is indicated for treating {entities}.",
    ),
}


def make_visit_turns(visit_idx: int, visit_date: str,
                     subject: str, entities: List[str], qtype: str) -> List[Dict[str, Any]]:
    patient_tpl, doctor_tpl = _TURN_TEMPLATES.get(qtype, _TURN_TEMPLATES["side_effect"])
    e_str = _fmt(entities)
    ctx = {"subject": subject, "entities": e_str, "entities_cap": e_str.capitalize()}
    return [
        {
            "speaker": "Patient",
            "text": patient_tpl.format(**ctx),
            "timestamp": f"{visit_date} 09:00:00",
            "dia_id": str(visit_idx * 2),
        },
        {
            "speaker": "Doctor",
            "text": doctor_tpl.format(**ctx),
            "timestamp": f"{visit_date} 09:05:00",
            "dia_id": str(visit_idx * 2 + 1),
        },
    ]


# ── QA pair generation ─────────────────────────────────────────────────────────

_QA_TEMPLATES: Dict[str, Dict[str, str]] = {
    "side_effect": {
        "complete":    "What side effects of {subject} has the patient reported across all visits?",
        "first_visit": "What side effect(s) did the patient first report when starting {subject}?",
        "latest":      "What side effect(s) of {subject} were most recently mentioned?",
    },
    "disease_symptom": {
        "complete":    "What symptoms has the patient mentioned across all consultations for {subject}?",
        "first_visit": "What symptom(s) did the patient first complain about?",
        "latest":      "What was the most recently reported symptom?",
    },
    "gene_bio_process": {
        "complete":    "What biological processes associated with the {subject} gene were identified in all reports?",
        "first_visit": "What biological process(es) of {subject} were noted in the first report?",
        "latest":      "What biological process(es) of {subject} were most recently documented?",
    },
    "gene_mol_function": {
        "complete":    "What molecular functions of the {subject} gene were identified across all analyses?",
        "first_visit": "What molecular function(s) of {subject} were first noted?",
        "latest":      "What molecular function(s) of {subject} were most recently reported?",
    },
    "gene_anatomy_down": {
        "complete":    "What anatomical structures can the {subject} gene downregulate, based on all reports?",
        "first_visit": "Which anatomical structure(s) were first mentioned as downregulated by {subject}?",
        "latest":      "Which anatomical structure(s) were most recently noted as downregulated by {subject}?",
    },
    "gene_anatomy_expr": {
        "complete":    "In which anatomical structures is the {subject} gene expressed, according to all reports?",
        "first_visit": "Which anatomical structure(s) were first noted for {subject} expression?",
        "latest":      "Which anatomical structure(s) were most recently noted for {subject} expression?",
    },
    "gene_anatomy_up": {
        "complete":    "What anatomical structures can the {subject} gene upregulate, based on all reports?",
        "first_visit": "Which anatomical structure(s) were first mentioned as upregulated by {subject}?",
        "latest":      "Which anatomical structure(s) were most recently upregulated by {subject}?",
    },
    "drug_compound": {
        "complete":    "What conditions can {subject} treat, according to our consultations?",
        "first_visit": "Which condition was first discussed as treatable with {subject}?",
        "latest":      "What was the most recently discussed condition treatable by {subject}?",
    },
    "drug_anatomy": {
        "complete":    "What condition is {subject} being used to treat in this patient's case?",
        "first_visit": "What was the first condition mentioned in relation to {subject}?",
        "latest":      "What condition was most recently discussed for {subject}?",
    },
}


def make_qa_pairs(session_id: str, subject: str, entities: List[str],
                  chunks: List[List[str]], qtype: str) -> List[Dict[str, Any]]:
    templates = _QA_TEMPLATES.get(qtype, _QA_TEMPLATES["side_effect"])
    ctx = {"subject": subject}
    qa_pairs = []

    # complete (cat=0) – always generated
    qa_pairs.append({
        "query_id": f"{session_id}_q0",
        "question": templates["complete"].format(**ctx),
        "ground_truth": ", ".join(entities),
        "category": 0,
        "meta": {"qa_type": "complete", "optimal_route": "R"},
    })

    # first_visit (cat=1) and latest (cat=2) – only when there are multiple chunks
    if len(chunks) > 1:
        qa_pairs.append({
            "query_id": f"{session_id}_q1",
            "question": templates["first_visit"].format(**ctx),
            "ground_truth": ", ".join(chunks[0]),
            "category": 1,
            "meta": {"qa_type": "first_visit", "optimal_route": "R"},
        })
        qa_pairs.append({
            "query_id": f"{session_id}_q2",
            "question": templates["latest"].format(**ctx),
            "ground_truth": ", ".join(chunks[-1]),
            "category": 2,
            "meta": {"qa_type": "latest", "optimal_route": "S"},
        })

    return qa_pairs


# ── Main conversion ────────────────────────────────────────────────────────────

def convert_qa_to_session(qa: Dict[str, Any]) -> Dict[str, Any]:
    qid = int(qa["qid"])
    question = qa["question"]
    answer = qa["answer"]
    qtype = get_qtype(qid)
    session_id = f"biomedical_{qid}"

    subject, entities = extract_subject_entities(question, answer, qtype)

    # Chunk entities: 2 per visit
    chunks = [entities[i:i + 2] for i in range(0, len(entities), 2)]
    n_visits = len(chunks)

    # Generate one visit date per chunk (16-day intervals)
    base = date(2024, 1, 15)
    visit_dates = [(base + timedelta(days=i * 16)).strftime("%Y-%m-%d") for i in range(n_visits)]

    # Build fine-grained turns and coarse-grained session_chunks
    all_turns: List[Dict[str, Any]] = []
    session_chunks: List[Dict[str, Any]] = []

    for visit_idx, (chunk, vdate) in enumerate(zip(chunks, visit_dates)):
        turns = make_visit_turns(visit_idx, vdate, subject, chunk, qtype)
        all_turns.extend(turns)

        chunk_lines = [f"=== VISIT {visit_idx + 1} - Date: {vdate} ==="]
        for t in turns:
            chunk_lines.append(f"{t['speaker']}: {t['text']}")
        session_chunks.append({
            "speaker": "system",
            "text": "\n".join(chunk_lines),
            "chunk_idx": visit_idx,
        })

    qa_pairs = make_qa_pairs(session_id, subject, entities, chunks, qtype)

    return {
        "session_id": session_id,
        "turns": all_turns,
        "session_chunks": session_chunks,
        "qa_pairs": qa_pairs,
        "meta": {
            "qid": qid,
            "qtype": qtype,
            "subject": subject,
            "original_question": question,
            "original_answer": answer,
            "num_entities": len(entities),
            "num_visits": n_visits,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert biomedical QA to med-memgate session format")
    parser.add_argument("--input",  default="core/datasets/biomedical/data.json",
                        help="Path to data.json")
    parser.add_argument("--output", default="core/datasets/biomedical/sessions.jsonl",
                        help="Path to output sessions.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sessions = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa = json.loads(line)
                sessions.append(convert_qa_to_session(qa))

    with open(output_path, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Stats
    total_qa = sum(len(s["qa_pairs"]) for s in sessions)
    total_turns = sum(len(s["turns"]) for s in sessions)
    multi_visit = sum(1 for s in sessions if s["meta"]["num_visits"] > 1)
    print(f"Converted {len(sessions)} QA pairs → {output_path}")
    print(f"  Total turns   : {total_turns}")
    print(f"  Total QA pairs: {total_qa}")
    print(f"  Multi-visit   : {multi_visit}/{len(sessions)} sessions")

    # Verify a sample
    print("\n--- Sample session (biomedical_0) ---")
    sample = sessions[0]
    print(f"  subject: {sample['meta']['subject']}")
    print(f"  visits : {sample['meta']['num_visits']}")
    print(f"  QA types: {[q['meta']['qa_type'] for q in sample['qa_pairs']]}")
    print(f"  Turn 0 : {sample['turns'][0]['text'][:80]}")


if __name__ == "__main__":
    main()
