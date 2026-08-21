"""The assistant itself: calls the right tool function and formats its
output as a FACTS/RECOMMENDATION answer. Also builds deliberately
flawed versions of that answer for the "rejected" side of DPO training
pairs, and contains the rule-based router that maps a raw question to a
tool call.

warehouse_id alone is ambiguous in this dataset - the same id is reused
across 3 different warehouse tiers (see data_utils.py) - so anything
that needs one specific warehouse uses a uid, and the router
disambiguates or reports why it couldn't.
"""

import random
import re

import data_utils as du

# not real category/reason values - deliberately outside the dataset's
# known 12 of each, used as negative examples for a fabricated category
# or fabricated reason code
_FAKE_CATEGORIES = ["general_supplies", "miscellaneous_items", "unclassified_stock"]
_FAKE_REASONS = ["warehouse_mismanagement", "unknown_cause", "system_error"]

# the real valid values (3 tiers, 3 region types) are a small closed set
# defined by the dataset schema, plus fake values used as negative
# examples the same way as the two lists above
_ALL_LEVELS = ["district_store", "national_CMS", "regional_warehouse"]
_ALL_REGIONS = ["urban", "peri_urban", "rural"]
_FAKE_LEVELS = ["provincial_store", "central_depot", "field_unit"]
_FAKE_REGIONS = ["suburban", "coastal", "highland"]

TOOLS = {
    "stockouts": "which items/warehouses currently have an active stockout",
    "backorders": "which items are backordered right now",
    "shortage_risk": "which warehouses/items have the highest shortage or replenishment risk",
    "kpis": "how a specific warehouse is performing overall (needs a warehouse id)",
    "explain_issue": "why a specific warehouse has an inventory issue (needs a warehouse id + item category)",
}

_WH_RE = re.compile(r"\bWH_\d+\b", re.IGNORECASE)

_LEVEL_SYNONYMS = {
    "district_store": ["district_store", "district store", "district"],
    "national_CMS": ["national_cms", "national cms", "central medical store", "national"],
    "regional_warehouse": ["regional_warehouse", "regional warehouse", "regional"],
}

# peri_urban is checked before urban, so "peri_urban" text doesn't also
# register as a separate "urban" match
_REGION_SYNONYMS_ORDERED = [
    ("peri_urban", ["peri_urban", "peri urban", "periurban", "peri-urban"]),
    ("urban", ["urban"]),
    ("rural", ["rural"]),
]

_TIER_CONCEPT_WORDS = ("tier", "tiers", "level", "levels")
_REGION_CONCEPT_WORDS = ("region", "regions")


def _format_group_facts(group_results, item_label_fn):
    """Formats a comparison result {group_value: tool_output} into one
    FACTS-line-friendly string, one segment per group, sorted for
    determinism."""
    parts = []
    for group_val in sorted(group_results.keys()):
        parts.append(f"{group_val}: {item_label_fn(group_results[group_val])}")
    return "; ".join(parts)


def _worst_group(group_results, metric_fn):
    """Finds which group is worst, by metric_fn applied to that group's
    result. Handles both shapes a tool function can return: a list
    (get_stockouts/get_backorders/rank_shortage_risk - sorted worst-first,
    so res[0] is the group's worst record) and a single dict
    (get_kpis/get_dominant_issue - one aggregate per group). Groups with
    nothing usable (empty list, or an insufficient-data dict) are
    skipped. Returns None if every group was empty or unusable."""
    candidates = []
    for g, res in group_results.items():
        if not res or (isinstance(res, dict) and res.get("status") == "insufficient_data"):
            continue
        target = res[0] if isinstance(res, list) else res
        candidates.append((g, metric_fn(target)))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def answer(records, intent, scope):
    """Calls the tool function matching intent, using scope to filter,
    scope a single warehouse, or compare across a group, and formats the
    real result into a FACTS/RECOMMENDATION answer. Returns (answer_text,
    raw_tool_output)."""
    if intent == "stockouts":
        if "compare_by" in scope:
            grouped = du.grouped(records, scope["compare_by"], du.get_stockouts,
                                  group_values=scope["compare_values"])
            label_fn = lambda res: (f"{len(res)} stockouts, worst {res[0]['warehouse_id']} "
                                     f"({res[0]['commodity_category']}, {res[0]['facilities_affected_by_stockout']} facilities)") if res else "no stockouts"
            facts = f"stockouts by {scope['compare_by']}: " + _format_group_facts(grouped["groups"], label_fn)
            worst = _worst_group(grouped["groups"], lambda top: top["facilities_affected_by_stockout"])
            rec = (f"prioritize the {worst} group first - most facilities affected."
                   if worst else f"no active stockouts in any compared {scope['compare_by']} group.")
            return f"FACTS: {facts}\nRECOMMENDATION: {rec}", grouped

        filters = {k: v for k, v in scope.items() if k in ("warehouse_level", "region_type", "uid")}
        res = du.get_stockouts(records, **filters)
        scope_label = f" ({', '.join(f'{k}={v}' for k, v in filters.items())})" if filters else ""
        if res:
            top = res[0]
            facts = f"{len(res)} active stockouts{scope_label}. worst: {top['warehouse_id']} ({top['warehouse_level']}, {top['commodity_category']}), {top['facilities_affected_by_stockout']} facilities hit"
            rec = f"prioritize {top['warehouse_id']} ({top['warehouse_level']}) first."
        else:
            facts = f"no active stockouts{scope_label}"
            rec = "nothing urgent here."
        return f"FACTS: {facts}\nRECOMMENDATION: {rec}", res

    if intent == "backorders":
        if "compare_by" in scope:
            grouped = du.grouped(records, scope["compare_by"], du.get_backorders,
                                  group_values=scope["compare_values"])
            label_fn = lambda res: (f"worst {res[0]['warehouse_id']} ({res[0]['commodity_category']}) at "
                                     f"{res[0]['backorder_rate_pct']}%, {res[0]['orders_backordered']} orders") if res else "none backordered"
            facts = f"backorders by {scope['compare_by']}: " + _format_group_facts(grouped["groups"], label_fn)
            worst = _worst_group(grouped["groups"], lambda top: top["backorder_rate_pct"])
            rec = (f"expedite the {worst} group first - highest backorder rate."
                   if worst else f"nothing backordered in any compared {scope['compare_by']} group.")
            return f"FACTS: {facts}\nRECOMMENDATION: {rec}", grouped

        filters = {k: v for k, v in scope.items() if k in ("warehouse_level", "region_type", "uid")}
        res = du.get_backorders(records, **filters)
        scope_label = f" ({', '.join(f'{k}={v}' for k, v in filters.items())})" if filters else ""
        if res:
            top = res[0]
            facts = f"{len(res)} backordered records{scope_label}. worst: {top['warehouse_id']} ({top['warehouse_level']}, {top['commodity_category']}) at {top['backorder_rate_pct']}% backorder rate"
            rec = f"expedite {top['warehouse_id']} ({top['warehouse_level']})."
        else:
            facts = f"nothing backordered right now{scope_label}"
            rec = "no action needed."
        return f"FACTS: {facts}\nRECOMMENDATION: {rec}", res

    if intent == "shortage_risk":
        if "compare_by" in scope:
            grouped = du.grouped(records, scope["compare_by"], du.rank_shortage_risk,
                                  group_values=scope["compare_values"], top_n=3)
            label_fn = lambda res: (f"top risk {res[0]['warehouse_id']} ({res[0]['commodity_category']}), "
                                     f"score={res[0]['risk_score']}") if res else "no elevated risk"
            facts = f"shortage risk by {scope['compare_by']}: " + _format_group_facts(grouped["groups"], label_fn)
            worst = _worst_group(grouped["groups"], lambda top: top["risk_score"])
            rec = (f"focus replenishment planning on the {worst} group first - highest risk score."
                   if worst else f"no elevated risk in any compared {scope['compare_by']} group.")
            return f"FACTS: {facts}\nRECOMMENDATION: {rec}", grouped

        filters = {k: v for k, v in scope.items() if k in ("warehouse_level", "region_type", "uid")}
        res = du.rank_shortage_risk(records, top_n=5, **filters)
        scope_label = f" ({', '.join(f'{k}={v}' for k, v in filters.items())})" if filters else ""
        if res:
            top = res[0]
            facts = f"top shortage risk{scope_label} is {top['warehouse_id']} ({top['warehouse_level']}, {top['commodity_category']}), score={top['risk_score']}, factors={top['risk_factors']}"
            rec = f"focus replenishment planning on {top['warehouse_id']} ({top['warehouse_level']})."
        else:
            facts = f"nothing looks risky right now{scope_label}"
            rec = "no action needed."
        return f"FACTS: {facts}\nRECOMMENDATION: {rec}", res

    if intent == "kpis":
        if "uid" in scope:
            res = du.get_warehouse_kpis(records, scope["uid"])
            if res is None:
                return "FACTS: no data for that warehouse\nRECOMMENDATION: can't recommend anything without data", None
            facts = f"{res['warehouse_id']} ({res['warehouse_level']}): avg inventory accuracy {res['avg_inventory_accuracy_pct']}%, avg fulfilment {res['avg_order_fulfilment_rate_pct']}%, {res['months_with_stockout']} months w/ stockout (based on {res['n_records']} records)"
            if res["issue_counts"]:
                facts += f", issues on record: {res['issue_counts']}"
            rec = "looks fine." if res["months_with_stockout"] == 0 else "worth a review."
            return f"FACTS: {facts}\nRECOMMENDATION: {rec}", res

        if "compare_by" in scope:
            grouped = du.grouped(records, scope["compare_by"], du.get_kpis, group_values=scope["compare_values"])
            label_fn = lambda res: (f"avg inventory accuracy {res['avg_inventory_accuracy_pct']}%, "
                                     f"avg fulfilment {res['avg_order_fulfilment_rate_pct']}%, "
                                     f"{res['months_with_stockout']} months w/ stockout") if res else "no data"
            facts = f"kpis by {scope['compare_by']}: " + _format_group_facts(grouped["groups"], label_fn)
            # months_with_stockout is the same "worst" signal the
            # single-warehouse case above uses to decide "looks fine" vs
            # "worth a review", kept consistent rather than inventing a
            # separate composite ranking
            worst = _worst_group(grouped["groups"], lambda top: top["months_with_stockout"])
            rec = (f"review the {worst} group first - most months with a stockout."
                   if worst else f"no data to compare across {scope['compare_by']} groups.")
            return f"FACTS: {facts}\nRECOMMENDATION: {rec}", grouped

        filters = {k: v for k, v in scope.items() if k in ("warehouse_level", "region_type")}
        res = du.get_kpis(records, **filters)
        scope_label = f" ({', '.join(f'{k}={v}' for k, v in filters.items())})" if filters else ""
        if res is None:
            return f"FACTS: no data{scope_label}\nRECOMMENDATION: can't recommend anything without data", None
        facts = (f"avg inventory accuracy {res['avg_inventory_accuracy_pct']}%, "
                 f"avg fulfilment {res['avg_order_fulfilment_rate_pct']}%, "
                 f"{res['months_with_stockout']} months w/ stockout{scope_label} (based on {res['n_records']} records)")
        if res["issue_counts"]:
            facts += f", issues on record: {res['issue_counts']}"
        rec = "looks fine." if res["months_with_stockout"] == 0 else "worth a review."
        return f"FACTS: {facts}\nRECOMMENDATION: {rec}", res

    if intent == "explain_issue":
        if "uid" in scope:
            res = du.explain_inventory_issue(records, scope["uid"], scope["commodity_category"],
                                              scope.get("year"), scope.get("month"))
            if res["status"] != "ok":
                return f"FACTS: {res['reason']}\nRECOMMENDATION: can't recommend anything without data", res
            signal_str = "; ".join(res["signals"]) if res["signals"] else "no other signals"
            facts = f"{res['warehouse_id']} ({res['warehouse_level']}, {res['month']}/{res['year']}): reported issue = {res['reported_issue']}. other signals: {signal_str}"
            if res.get("note"):
                facts += f" ({res['note']})"
            elif res.get("uncertain"):
                facts += " (note: report not submitted, numbers may be provisional)"
            rec = f"address the reported issue ({res['reported_issue']})." if res["has_issue"] else "no action needed."
            return f"FACTS: {facts}\nRECOMMENDATION: {rec}", res

        if "compare_by" in scope:
            grouped = du.grouped(records, scope["compare_by"], du.get_dominant_issue,
                                  group_values=scope["compare_values"])
            label_fn = lambda res: (f"{res['dominant_issue']} ({res['count']}x)"
                                     if res and res.get("status") != "insufficient_data" else "no recorded issues")
            facts = f"most common issue by {scope['compare_by']}: " + _format_group_facts(grouped["groups"], label_fn)
            worst = _worst_group(grouped["groups"], lambda top: top["count"])
            rec = (f"focus on the {worst} group's most common issue first."
                   if worst else f"no recorded issues in any compared {scope['compare_by']} group.")
            return f"FACTS: {facts}\nRECOMMENDATION: {rec}", grouped

        # aggregate, not tied to one record: reports the most common real
        # issue code among matching rows, not a guess or an average - see
        # get_dominant_issue()'s docstring for why this is the only
        # grounded aggregate answer at tier/region scope
        filters = {k: v for k, v in scope.items() if k in ("warehouse_level", "region_type", "commodity_category")}
        res = du.get_dominant_issue(records, **filters)
        scope_label = f" ({', '.join(f'{k}={v}' for k, v in filters.items())})" if filters else ""
        if res.get("status") == "insufficient_data":
            return f"FACTS: no recorded issues{scope_label}\nRECOMMENDATION: nothing to flag", res
        facts = f"most common issue{scope_label}: {res['dominant_issue']} ({res['count']} of {res['n_records']} records)"
        rec = f"investigate {res['dominant_issue']} as the top recurring issue."
        return f"FACTS: {facts}\nRECOMMENDATION: {rec}", res

    raise ValueError("unknown intent " + intent)


def break_it(good_response, scope, rng, uncertain=False):
    """Builds a deliberately flawed version of a correct answer, for the
    "rejected" side of a DPO training pair.

    Structural-only flaws (looping or redundant repetition) are never
    the sole reason a response is rejected here - they only ever
    compound on top of an already-substantive flaw (a wrong fact,
    a fabricated entity, or an unwarranted hedge), at a capped rate, so
    that structure alone can never fully account for a response being
    rejected.

    For prompts where real data is available (uncertain=False), one of
    three hedge-shaped rejected responses may also be selected with
    elevated probability, since an inappropriate hedge on an answerable
    question is a real, common failure mode worth strong negative
    coverage:
      - no_answer: a full, generic refusal.
      - contradicting_recommendation: the real facts stay correct and
        intact, but the recommendation hedges anyway - weighted highest,
        as the most representative shape of this failure.
      - hedge_in_facts: a hedge phrase embedded within the facts
        section, with an otherwise normal recommendation.
    """
    major_issue_modes = ["wrong_warehouse", "fact_is_rec", "wrong_number", "no_answer",
              "fake_category", "fake_reason",
              "wrong_grouping_dim", "swapped_dimension", "dropped_group",
              "fake_group_value", "repeated_number"]

    broken = None
    if not uncertain and rng.random() < 0.5:
        hedge_mode = rng.choices(
            ["no_answer", "contradicting_recommendation", "hedge_in_facts"],
            weights=[1, 3, 1],
        )[0]
        b = _break(good_response, scope, hedge_mode, rng)
        if b != good_response:
            broken = b

    if broken is None:
        rng.shuffle(major_issue_modes)
        broken = "not sure, you'd have to check the system yourself"
        for mode in major_issue_modes:
            b = _break(good_response, scope, mode, rng)
            if b != good_response:
                broken = b
                break

    # a structural flaw only ever gets added on top of the major issue
    # already selected above, and only some of the time - if the step
    # above fell through to the generic refusal (no FACTS/RECOMMENDATION
    # structure or digits to loop/repeat), this safely no-ops via
    # _break()'s own unchanged-text check
    if rng.random() < 0.25:
        oq_mode = rng.choice(["looping_response", "redundant_repeat"])
        compounded = _break(broken, scope, oq_mode, rng)
        if compounded != broken:
            broken = compounded

    return broken


def _parse_compare_facts(text):
    """Extracts (prefix, dim, [(group_label, segment_text), ...]) from
    the fixed 'FACTS: X by DIM: g1: s1; g2: s2; ...\\nRECOMMENDATION:'
    shape answer() always uses for compare-type responses. Returns None
    if text doesn't match that shape (e.g. it's a plain filter/unscoped
    answer) - every flaw mode below relies on this to safely no-op on
    non-compare text."""
    m = re.match(r"FACTS: (.+?) by (\w+): (.+?)\nRECOMMENDATION:", text, re.S)
    if not m:
        return None
    prefix, dim, body = m.group(1), m.group(2), m.group(3)
    segments = []
    for part in body.split("; "):
        if ": " not in part:
            continue
        label, seg_text = part.split(": ", 1)
        segments.append((label, seg_text))
    return prefix, dim, segments


def _rebuild_compare_facts(text, prefix, dim, segments):
    """Reassembles a compare-shaped FACTS/RECOMMENDATION string from
    parsed parts, keeping the original recommendation text untouched."""
    rec_part = text.split("RECOMMENDATION:")[-1]
    body = "; ".join(f"{label}: {seg}" for label, seg in segments)
    return f"FACTS: {prefix} by {dim}: {body}\nRECOMMENDATION:{rec_part}"


def _break(text, scope, mode, rng):
    """Applies one specific flaw to text, or returns it unchanged if
    that flaw's precondition isn't met (e.g. a compare-only flaw applied
    to a single-answer response)."""
    if mode == "wrong_warehouse":
        m = re.search(r"\bWH_\d+\b", text)
        if m:
            fake = f"WH_{rng.randint(100,899):04d}"
            return text.replace(m.group(0), fake)
        return text

    if mode == "fact_is_rec":
        rec = text.split("RECOMMENDATION:")[-1].strip()
        return "FACTS: " + rec  # states the recommendation as if it were a fact

    if mode == "wrong_number":
        m = re.search(r"(?<!_)\b\d+\.?\d*\b(?!_)", text)
        if not m:
            return text
        orig = m.group()
        newval = str(round(float(orig) + rng.uniform(15, 40), 1))
        return re.sub(rf"(?<!_)\b{re.escape(orig)}\b(?!_)", newval, text, count=1)

    if mode == "fake_category":
        # only applies to the "(LEVEL, CATEGORY)" pattern used by
        # stockouts/backorders/shortage_risk
        m = re.search(r"\(([a-zA-Z_]+),\s*([a-zA-Z_]+)\)", text)
        if not m:
            return text
        fake = rng.choice(_FAKE_CATEGORIES)
        return text[:m.start(2)] + fake + text[m.end(2):]

    if mode == "fake_reason":
        # matches both explain_issue phrasings: the single-warehouse
        # "reported issue = X" and the tier/region/compare aggregate's
        # "most common issue...: X ("
        m = re.search(r"reported issue = ([a-zA-Z_]+)", text)
        if not m:
            m = re.search(r"most common issue[^:]*:\s*([a-zA-Z_]+)", text)
        if not m:
            return text
        fake = rng.choice(_FAKE_REASONS)
        return text[:m.start(1)] + fake + text[m.end(1):]

    if mode == "looping_response":
        # restates the whole answer under a fresh label instead of
        # finishing it, rather than substituting or removing something
        # in the existing structure. works on any response shape since
        # it only splits on the RECOMMENDATION: marker
        if "RECOMMENDATION:" not in text:
            return text
        facts_part = text.split("\nRECOMMENDATION:")[0]
        rec_part = text.split("RECOMMENDATION:")[-1].strip()
        return f"{text}\n{facts_part}\nRECOMMENDATION: {rec_part}"

    if mode == "redundant_repeat":
        # restates one real number two more times in added filler
        # sentences, appended rather than edited into the existing text
        m = re.search(r"(?<!_)\b\d+\.?\d*\b(?!_)", text)
        if not m:
            return text
        num = m.group()
        return f"{text}\nFACTS: to repeat, the figure is {num}. Again, {num} is the number."

    # the 5 modes below only apply to compare-type responses (the ones
    # with "compare_by" in scope) - they all safely no-op on any other
    # response shape via _parse_compare_facts returning None

    if mode == "wrong_grouping_dim":
        # asked to compare tiers, answered with individual warehouses
        # instead - same numbers, wrong axis entirely
        parsed = _parse_compare_facts(text)
        if not parsed:
            return text
        prefix, dim, segments = parsed
        fake_segments = [(f"WH_{rng.randint(100,899):04d}", seg) for _, seg in segments]
        return _rebuild_compare_facts(text, prefix, "warehouse", fake_segments)

    if mode == "swapped_dimension":
        # asked to compare tiers, answered with region-type groups
        # instead (or vice versa) - both dimensions have 3 values each,
        # making this a plausible mixup
        parsed = _parse_compare_facts(text)
        if not parsed or "compare_by" not in scope:
            return text
        prefix, dim, segments = parsed
        is_level = scope["compare_by"] == "warehouse_level"
        other_values = _ALL_REGIONS if is_level else _ALL_LEVELS
        other_dim = "region_type" if is_level else "warehouse_level"
        if len(other_values) < len(segments):
            return text
        fake_segments = [(other_values[i], seg) for i, (_, seg) in enumerate(segments)]
        return _rebuild_compare_facts(text, prefix, other_dim, fake_segments)

    if mode == "dropped_group":
        # silently omits one group from an otherwise-correct comparison -
        # nothing stated is factually wrong, it just misrepresents the
        # comparison as exhaustive when it isn't
        parsed = _parse_compare_facts(text)
        if not parsed or len(parsed[2]) < 2:
            return text
        prefix, dim, segments = parsed
        idx = rng.randrange(len(segments))
        new_segments = segments[:idx] + segments[idx + 1:]
        return _rebuild_compare_facts(text, prefix, dim, new_segments)

    if mode == "fake_group_value":
        # invents a tier/region that doesn't exist in the data at all
        parsed = _parse_compare_facts(text)
        if not parsed or not parsed[2]:
            return text
        prefix, dim, segments = parsed
        idx = rng.randrange(len(segments))
        fake_val = rng.choice(_FAKE_LEVELS if dim == "warehouse_level" else _FAKE_REGIONS)
        new_segments = list(segments)
        new_segments[idx] = (fake_val, segments[idx][1])
        return _rebuild_compare_facts(text, prefix, dim, new_segments)

    if mode == "repeated_number":
        # correct group labels, but the same pooled figure copied under
        # every group instead of each group's real, distinct number -
        # looks like a real comparison but misrepresents the
        # differentiation that was actually asked for
        parsed = _parse_compare_facts(text)
        if not parsed or len(parsed[2]) < 2:
            return text
        prefix, dim, segments = parsed
        first_seg_text = segments[0][1]
        new_segments = [(label, first_seg_text) for label, _ in segments]
        return _rebuild_compare_facts(text, prefix, dim, new_segments)

    if mode == "contradicting_recommendation":
        # keeps the real facts intact and correct, but replaces the
        # recommendation with an unwarranted hedge
        if "\nRECOMMENDATION:" not in text:
            return text
        facts_part = text.split("\nRECOMMENDATION:")[0]
        hedge_recs = [
            "Not sure. (Not enough information to determine)",
            "Not sure. (Not enough information)",
            "Not sure. To get more information, I would need to check further.",
            "Not sure about that - would need to check the system for more detail.",
        ]
        return f"{facts_part}\nRECOMMENDATION: {rng.choice(hedge_recs)}"

    if mode == "hedge_in_facts":
        # real facts stay, but a hedge phrase gets appended within the
        # facts section itself, while the recommendation stays normal
        if "\nRECOMMENDATION:" not in text:
            return text
        facts_part, rec_part = text.split("\nRECOMMENDATION:", 1)
        hedge_additions = [
            " Not sure about the rest of the details.",
            " Not entirely sure if this is complete.",
            " Not sure if there's more to report here.",
        ]
        return f"{facts_part}{rng.choice(hedge_additions)}\nRECOMMENDATION:{rec_part}"

    # no_answer
    return "not sure, you'd have to check the system yourself"


def extract_warehouse_id(text):
    """Returns the first "WH_####"-shaped id found in text, normalized
    to uppercase, or None."""
    m = _WH_RE.search(text)
    if not m:
        return None
    raw = m.group(0).upper()
    return raw.replace("wh_", "WH_")


def extract_level_hint(text):
    """Returns the first warehouse tier mentioned in text, or None."""
    tl = text.lower()
    for level, synonyms in _LEVEL_SYNONYMS.items():
        for syn in synonyms:
            if syn in tl:
                return level
    return None


def extract_level_hints(text):
    """All distinct tier mentions found, in the order they appear -
    the all-matches version of extract_level_hint, used for group-by
    scoping (filter vs. compare depends on how many are named)."""
    tl = text.lower()
    found = []
    for level, synonyms in _LEVEL_SYNONYMS.items():
        for syn in synonyms:
            if syn in tl:
                found.append(level)
                break
    return found


def extract_region_hints(text):
    """All distinct region-type mentions found, in the order they
    appear - same idea as extract_level_hints, for region_type."""
    tl = text.lower()
    remaining = tl
    found = []
    for region, synonyms in _REGION_SYNONYMS_ORDERED:
        for syn in synonyms:
            if syn in remaining:
                found.append(region)
                remaining = remaining.replace(syn, "", 1)
                break
    return found


def resolve_group_scope(text, concept_words, hint_fn, all_values):
    """Decides how one scoping dimension (tier or region) applies to a
    question:
      0 names, no concept word  -> ('none', None)            - unscoped
      0 names, concept word     -> ('compare', all_values)   - "compare tiers"
      1 name                    -> ('filter', that_value)    - "in the district tier"
      2+ names                  -> ('compare', those_values) - "district vs national"

    concept_words is matched with a word-boundary regex rather than
    plain substring containment, since a tier name can itself contain a
    region-related substring (e.g. "regional_warehouse" contains
    "region"), which a naive substring check would misread as a mention
    of the region dimension.
    """
    tl = text.lower()
    hints = hint_fn(text)
    unique_hints = list(dict.fromkeys(hints))  # dedupe, keep first-seen order
    if len(unique_hints) == 1:
        return "filter", unique_hints[0]
    if len(unique_hints) >= 2:
        return "compare", unique_hints
    if any(re.search(rf"\b{re.escape(w)}\b", tl) for w in concept_words):
        return "compare", sorted(all_values)
    return "none", None


def resolve_dataset_wide_scope(text, records):
    """Combines the tier and region resolution for the 3 dataset-wide
    tools (stockouts/backorders/shortage_risk) into one scope dict.

    A specific warehouse mention (e.g. "at WH_0009") takes priority over
    any tier/region language in the same text, since a specific
    warehouse already implies one fixed tier and region - further
    filtering on top would be redundant at best. If the id is present
    but ambiguous or unknown, this returns {"reason": <why>} instead of
    falling back to tier/region parsing.

    Filter combines with filter on the other dimension ("district tier
    in rural areas" applies both together), but compare on either side
    does not combine with anything else - two-dimensional grouping (e.g.
    "compare tiers within rural areas") isn't supported; compare wins
    over an unresolved filter on the other dimension.
    """
    if extract_warehouse_id(text):
        uid, err = resolve_warehouse(text, records)
        if err:
            return {"reason": err}
        return {"uid": uid}

    levels, regions = du.valid_levels_and_regions(records)
    level_mode, level_val = resolve_group_scope(text, _TIER_CONCEPT_WORDS, extract_level_hints, levels)
    region_mode, region_val = resolve_group_scope(text, _REGION_CONCEPT_WORDS, extract_region_hints, regions)

    if level_mode == "filter" and region_mode == "filter":
        return {"warehouse_level": level_val, "region_type": region_val}

    if level_mode == "compare" and region_mode != "none":
        region_mode, region_val = "none", None  # compare wins over an unresolved region scope
    elif region_mode == "compare" and level_mode != "none":
        level_mode, level_val = "none", None  # compare wins over an unresolved level scope

    if level_mode == "filter":
        return {"warehouse_level": level_val}
    if level_mode == "compare":
        return {"compare_by": "warehouse_level", "compare_values": level_val}
    if region_mode == "filter":
        return {"region_type": region_val}
    if region_mode == "compare":
        return {"compare_by": "region_type", "compare_values": region_val}
    return {}


def extract_commodity(text, records):
    """Returns the first commodity category mentioned in text, matching
    either the raw stored value or its space-separated form, or None."""
    cats = {r["commodity_category"] for r in records}
    tl = text.lower()
    for cat in cats:
        if cat.replace("_", " ").lower() in tl or cat.lower() in tl:
            return cat
    return None


def resolve_warehouse(text, records):
    """Resolves a warehouse id mentioned in text to one specific uid.
    Returns (uid, error_reason); error_reason is None on success."""
    wh = extract_warehouse_id(text)
    if not wh:
        return None, "need a warehouse id for this"

    candidates = du.find_uids_for_bare_id(records, wh)
    if not candidates:
        return None, f"no warehouse {wh} found in the data"

    level = extract_level_hint(text)
    if level:
        matching = [uid for uid, lvl in candidates if lvl == level]
        if not matching:
            return None, f"no record for {wh} at the {level} level"
        return matching[0], None

    if len(candidates) > 1:
        levels = ", ".join(lvl for _, lvl in candidates)
        return None, f"{wh} is ambiguous - matches {len(candidates)} warehouses ({levels}) - say which level"

    return candidates[0][0], None


def _route_dataset_wide(intent, text, records):
    """Wraps resolve_dataset_wide_scope() for the dataset-wide-capable
    intents, converting an unresolvable warehouse mention (ambiguous or
    unknown id) into routing failure rather than a scope dict with a
    stray "reason" key that answer()/the tool functions would choke
    on."""
    scope = resolve_dataset_wide_scope(text, records)
    if "reason" in scope:
        return None, scope
    return intent, scope


def route(text, records):
    """Maps a raw question to (intent, scope). intent is None if
    nothing matched confidently, or a required piece of scope couldn't
    be resolved - scope['reason'] explains why in that case. This is a
    plain keyword classifier, not the model itself doing function
    calling.

    Match order here is deliberate: risk/ranking language is checked
    before the narrower "stockout" check, since "stockout risk" and
    "riskiest warehouses" are asking for a ranking, not a list of
    current stockouts.
    """
    tl = text.lower()
    has_wh_id = extract_warehouse_id(text) is not None

    if "backorder" in tl or "overdue" in tl:
        return _route_dataset_wide("backorders", text, records)

    # causal/explain language ("why", "struggling", etc) means
    # explain_issue only when a specific target is named - a warehouse
    # id ("why is WH_0009 struggling"), or a specific named tier/region
    # ("why is national_CMS struggling"). "what warehouse is struggling
    # the most" (nothing specific named) is a ranking question and
    # belongs to shortage_risk below instead.
    explain_words = ("why", "failing", "explain", "issue", "struggling")
    if has_wh_id and any(w in tl for w in explain_words):
        uid, err = resolve_warehouse(text, records)
        if err:
            return None, {"reason": err}
        cat = extract_commodity(text, records)
        if not cat:
            return None, {"reason": "need an item category for this"}
        return "explain_issue", {"uid": uid, "commodity_category": cat}

    if not has_wh_id and any(w in tl for w in explain_words):
        scope = resolve_dataset_wide_scope(text, records)
        if ("warehouse_level" in scope or "region_type" in scope) and "compare_by" not in scope:
            cat = extract_commodity(text, records)
            if cat:
                scope["commodity_category"] = cat
            return "explain_issue", scope

    risk_words = ("shortage", "risk", "urgent", "replenish", "struggling",
                  "running out", "run out", "attention")
    if any(w in tl for w in risk_words):
        return _route_dataset_wide("shortage_risk", text, records)

    if "stockout" in tl or "stocked out" in tl or "stock out" in tl:
        return _route_dataset_wide("stockouts", text, records)

    if not has_wh_id and any(w in tl for w in explain_words):
        # reaches here if explain_words matched but no clean single
        # tier/region filter was found above - either fully unscoped
        # ("why is fulfillment failing?") or compare-shaped ("why do
        # tiers differ?"). resolved the same way as the other
        # dataset-wide intents; category is optional here, unlike the
        # single-warehouse case above, since a tier-wide "why" question
        # is answerable across all categories at once.
        scope = resolve_dataset_wide_scope(text, records)
        if "reason" in scope:
            return None, scope
        cat = extract_commodity(text, records)
        if cat:
            scope["commodity_category"] = cat
        return "explain_issue", scope

    if "how is" in tl or "doing overall" in tl or "kpi" in tl or "performance" in tl:
        return _route_dataset_wide("kpis", text, records)

    return None, {"reason": "couldn't tell which of the 5 tools this question needs"}


def answer_from_text(records, text):
    """The full entry point: a raw question in, a routed and grounded
    answer out."""
    intent, scope = route(text, records)
    if intent is None:
        return f"FACTS: not sure what you're asking - {scope['reason']}\nRECOMMENDATION: rephrase with a warehouse id (like WH_0009) and level if you have one", None
    return answer(records, intent, scope)
