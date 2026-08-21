"""Builds the prompt list, generates grounded/broken answer pairs,
scores them, and picks chosen/rejected for DPO training.

Question text always embeds the warehouse level along with the id (e.g.
"WH_0009 at the national_CMS"), not just the bare id, since the same id
is reused across 3 different real warehouses (see data_utils.py) and
would otherwise be ambiguous.
"""

import itertools
import random
import re

import assistant as A
import data_utils as du

# shared generation length default, used by every entry point that
# generates from a model (eval_model.py, run.py, try_model.py)
DEFAULT_MAX_NEW = 100

SYS = ("You're a warehouse assistant. Answer using only the data given. "
       "Separate FACTS from RECOMMENDATION. Say if you're not sure. Don't make stuff up.")

# a couple question templates per tool
TOOL_QS = {
    "stockouts": ["which items are stocked out right now?", "any active stockouts?"],
    "backorders": ["which items are backordered?", "what's our backorder situation?"],
    "shortage_risk": ["which warehouses have the highest shortage risk?", "what needs urgent replenishment?"],
    "kpis": ["how is {wh} doing overall?"],
    "explain_issue": ["why is {wh} failing fulfillment for {cat}?"],
}

# single-warehouse-scoped versions of the 3 dataset-wide tools, e.g.
# "which category has the most backorders at WH_0009?"
TOOL_QS_SINGLE_WH = {
    "stockouts": "does {wh} currently have any active stockouts?",
    "backorders": "which category has the most backorders at {wh}?",
    "shortage_risk": "is {wh} at risk of running short on anything?",
}

# filter/compare templates for the 3 dataset-wide tools, across the two
# scoping dimensions (warehouse_level, region_type) - covers all 4 scope
# shapes route()/resolve_dataset_wide_scope() support: unscoped (above),
# single-dimension filter, compare-all, compare-named
_DATASET_WIDE_INTENTS = ["stockouts", "backorders", "shortage_risk"]

TIER_FILTER_QS = {
    "stockouts": "which items are stocked out in the {level} tier?",
    "backorders": "which items are backordered in the {level} tier?",
    "shortage_risk": "what needs urgent replenishment in the {level} tier?",
    "kpis": "what are the kpis for the {level} tier?",
    "explain_issue": "why are warehouses in the {level} tier failing?",
}
REGION_FILTER_QS = {
    "stockouts": "which items are stocked out in {region} areas?",
    "backorders": "which items are backordered in {region} areas?",
    "shortage_risk": "what needs urgent replenishment in {region} areas?",
    "kpis": "what are the kpis for {region} areas?",
    "explain_issue": "why are warehouses in {region} areas failing?",
}
TIER_COMPARE_ALL_QS = {
    "stockouts": "compare stockouts across tiers",
    "backorders": "compare backorder rates across tiers",
    "shortage_risk": "compare shortage risk across tiers",
    "kpis": "compare kpis across tiers",
    "explain_issue": "compare why warehouses are failing across tiers",
}
REGION_COMPARE_ALL_QS = {
    "stockouts": "compare stockouts across region types",
    "backorders": "compare backorder rates across region types",
    "shortage_risk": "compare shortage risk across region types",
    "kpis": "compare kpis across region types",
    "explain_issue": "compare why warehouses are failing across region types",
}
TIER_COMPARE_NAMED_QS = {
    "stockouts": "how do {a} and {b} compare on stockouts?",
    "backorders": "how do {a} and {b} compare on backorders?",
    "shortage_risk": "how do {a} and {b} compare on shortage risk?",
    "kpis": "how do {a} and {b} compare on kpis?",
    "explain_issue": "how do {a} and {b} compare on why they're failing?",
}
REGION_COMPARE_NAMED_QS = {
    "stockouts": "how do {a} and {b} compare on stockouts?",
    "backorders": "how do {a} and {b} compare on backorders?",
    "shortage_risk": "how do {a} and {b} compare on shortage risk?",
    "kpis": "how do {a} and {b} compare on kpis?",
    "explain_issue": "how do {a} and {b} compare on why they're failing?",
}
UNSCOPED_QS = {
    "kpis": ["what are the overall kpis?", "how's everything doing overall?"],
    "explain_issue": ["what's the most common issue overall?", "why do warehouses typically fail?"],
}

# single-record questions, for volume
FIELD_QS = {
    "fulfilment": (["what was the fulfilment rate at {wh} for {cat} in {mo}/{yr}?"],
                   ["order_fulfilment_rate_pct", "orders_backordered"]),
    "accuracy": (["what's the inventory accuracy at {wh} for {cat}?"],
                 ["inventory_accuracy_pct", "stock_record_up_to_date"]),
    "storage": (["are storage conditions ok at {wh} for {cat}?"],
                ["storage_conditions_adequate", "temperature_excursion_month"]),
    "wastage": (["what's the wastage rate at {wh} for {cat}?"],
                ["wastage_rate_pct", "expired_stock_value_usd"]),
}


def humanize(cat):
    """Turns a stored category value ("essential_medicines") into
    readable text ("essential medicines")."""
    return cat.replace("_", " ")


def wh_label(warehouse_id, level):
    """Formats a warehouse id with its tier, so question text is
    unambiguous about which of the (possibly 3) same-id warehouses is
    meant."""
    return f"{warehouse_id} at the {level}"


def _stratified_warehouse_sample(by_uid, target_n, rng):
    """Picks target_n warehouse uids, maximizing (a) proportional
    coverage across all warehouse tiers and (b) coverage of as many
    distinct real inventory_issue types as possible within each tier's
    share of the sample. A plain uniform-random sample of this size
    would likely still cover most tiers/issues by chance, but doesn't
    guarantee it - a rare issue type could be missed entirely, which
    would weaken the no_invented_reason/no_invented_category checks for
    anything they never saw a real example of.

    Deterministic given the seeded rng (shuffles before greedily
    picking, rather than relying on CSV row order).
    """
    by_level = {}
    for uid, rows in by_uid.items():
        by_level.setdefault(rows[0]["warehouse_level"], []).append(uid)

    levels = sorted(by_level.keys())
    # split target proportionally across tiers - floor division plus
    # distributing the remainder across the first few tiers
    # (alphabetical, for determinism)
    base, remainder = divmod(target_n, len(levels))
    per_level_target = {level: base for level in levels}
    for level in levels[:remainder]:
        per_level_target[level] += 1

    selected = []
    for level in levels:
        uids = list(by_level[level])
        rng.shuffle(uids)
        n_want = min(per_level_target[level], len(uids))

        covered_issues = set()
        picked = []
        remaining = list(uids)
        # greedy pass: prefer warehouses that introduce at least one
        # issue type not yet covered within this tier's picks
        for uid in list(remaining):
            if len(picked) >= n_want:
                break
            issues = {r["inventory_issue"] for r in by_uid[uid] if r["inventory_issue"] != "none"}
            if issues - covered_issues:
                picked.append(uid)
                covered_issues |= issues
                remaining.remove(uid)
        # fill any leftover slots with whatever's left (still
        # deterministic, since `remaining` was already shuffled above)
        for uid in remaining:
            if len(picked) >= n_want:
                break
            picked.append(uid)

        selected.extend(picked)

    return selected


def build_prompts(records, min_prompts=100, seed=42):
    """Builds the full list of question prompts: fixed tool-level
    questions, filter/compare questions across tier and region for
    every intent, single-warehouse questions for a stratified sample of
    warehouses, and single-field questions to pad out to min_prompts if
    needed."""
    rng = random.Random(seed)
    prompts = []
    qid = 0

    for intent in ["stockouts", "backorders", "shortage_risk"]:
        for q in TOOL_QS[intent]:
            qid += 1
            prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": intent, "scope": {}})

    by_uid = {}
    for r in records:
        by_uid.setdefault(r["uid"], []).append(r)

    # filter/compare coverage for the 3 dataset-wide tools - one real
    # example of each scope shape per intent, using whatever tiers/
    # regions actually exist in this dataset (not hardcoded)
    levels, regions = du.valid_levels_and_regions(records)
    levels, regions = sorted(levels), sorted(regions)
    for intent in _DATASET_WIDE_INTENTS:
        for level in levels:
            qid += 1
            q = TIER_FILTER_QS[intent].format(level=level)
            prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": intent,
                             "scope": {"warehouse_level": level}})
        for region in regions:
            qid += 1
            q = REGION_FILTER_QS[intent].format(region=region)
            prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": intent,
                             "scope": {"region_type": region}})

        qid += 1
        prompts.append({"qid": f"T{qid}", "text": TIER_COMPARE_ALL_QS[intent], "family": "tool", "intent": intent,
                         "scope": {"compare_by": "warehouse_level", "compare_values": levels}})
        qid += 1
        prompts.append({"qid": f"T{qid}", "text": REGION_COMPARE_ALL_QS[intent], "family": "tool", "intent": intent,
                         "scope": {"compare_by": "region_type", "compare_values": regions}})

        for a, b in itertools.combinations(levels, 2):
            qid += 1
            q = TIER_COMPARE_NAMED_QS[intent].format(a=a, b=b)
            prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": intent,
                             "scope": {"compare_by": "warehouse_level", "compare_values": [a, b]}})
        for a, b in itertools.combinations(regions, 2):
            qid += 1
            q = REGION_COMPARE_NAMED_QS[intent].format(a=a, b=b)
            prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": intent,
                             "scope": {"compare_by": "region_type", "compare_values": [a, b]}})

    # same 4 scope shapes for kpis/explain_issue - kept as its own loop
    # rather than merged into _DATASET_WIDE_INTENTS above, since these
    # two already get exhaustive per-warehouse coverage below and
    # merging would duplicate that coverage
    for intent in ["kpis", "explain_issue"]:
        for q in UNSCOPED_QS[intent]:
            qid += 1
            prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": intent,
                             "scope": {}})
        for level in levels:
            qid += 1
            q = TIER_FILTER_QS[intent].format(level=level)
            prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": intent,
                             "scope": {"warehouse_level": level}})
        for region in regions:
            qid += 1
            q = REGION_FILTER_QS[intent].format(region=region)
            prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": intent,
                             "scope": {"region_type": region}})

        qid += 1
        prompts.append({"qid": f"T{qid}", "text": TIER_COMPARE_ALL_QS[intent], "family": "tool", "intent": intent,
                         "scope": {"compare_by": "warehouse_level", "compare_values": levels}})
        qid += 1
        prompts.append({"qid": f"T{qid}", "text": REGION_COMPARE_ALL_QS[intent], "family": "tool", "intent": intent,
                         "scope": {"compare_by": "region_type", "compare_values": regions}})

        for a, b in itertools.combinations(levels, 2):
            qid += 1
            q = TIER_COMPARE_NAMED_QS[intent].format(a=a, b=b)
            prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": intent,
                             "scope": {"compare_by": "warehouse_level", "compare_values": [a, b]}})
        for a, b in itertools.combinations(regions, 2):
            qid += 1
            q = REGION_COMPARE_NAMED_QS[intent].format(a=a, b=b)
            prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": intent,
                             "scope": {"compare_by": "region_type", "compare_values": [a, b]}})

    # single-warehouse-scoped coverage, shared across all 5 intents - one
    # stratified sample of warehouses (see _stratified_warehouse_sample),
    # reused for every intent so "which warehouses do we cover" is one
    # decision instead of five. target 80 per intent: 16 fixed
    # structural prompts above (2 unscoped + 3 tier filter + 3 region
    # filter + 1 tier-compare-all + 1 region-compare-all + 3
    # tier-compare-named + 3 region-compare-named) plus 64 sampled
    # single-warehouse prompts = 80 per intent, 400 total.
    TARGET_PER_INTENT = 80
    n_fixed_per_intent = 16
    sample_size = TARGET_PER_INTENT - n_fixed_per_intent
    sample_uids = _stratified_warehouse_sample(by_uid, min(sample_size, len(by_uid)), rng)

    for uid in sample_uids:
        wh, level = by_uid[uid][0]["warehouse_id"], by_uid[uid][0]["warehouse_level"]
        label = wh_label(wh, level)
        for intent in _DATASET_WIDE_INTENTS:
            qid += 1
            q = TOOL_QS_SINGLE_WH[intent].format(wh=label)
            prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": intent,
                             "scope": {"uid": uid}})

        qid += 1
        q = TOOL_QS["kpis"][0].format(wh=label)
        prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": "kpis",
                         "scope": {"uid": uid}})

        rows = by_uid[uid]
        r = rng.choice(rows)
        qid += 1
        q = TOOL_QS["explain_issue"][0].format(wh=label, cat=humanize(r["commodity_category"]))
        prompts.append({"qid": f"T{qid}", "text": q, "family": "tool", "intent": "explain_issue",
                         "scope": {"uid": uid, "commodity_category": r["commodity_category"],
                                   "year": r["year"], "month": r["month"]}})

    # single field prompts, cycling through records + intents, repeating
    # with more rounds until min_prompts is reached
    field_names = list(FIELD_QS.keys())
    extra_round = 0
    while len(prompts) < min_prompts:
        for i, r in enumerate(records):
            intent = field_names[i % len(field_names)]
            templates, fields = FIELD_QS[intent]
            label = wh_label(r["warehouse_id"], r["warehouse_level"])
            q = templates[0].format(wh=label, cat=humanize(r["commodity_category"]),
                                     mo=r["month"], yr=r["year"])
            qid += 1
            prompts.append({
                "qid": f"S{qid}-{extra_round}", "text": q, "family": "field", "intent": intent,
                "scope": {"uid": r["uid"], "commodity_category": r["commodity_category"],
                          "year": r["year"], "month": r["month"]},
                "fields": fields,
            })
            if len(prompts) >= min_prompts:
                break
        extra_round += 1
        if extra_round > 50:
            break  # safety valve
    return prompts


def find_record(records, scope):
    """Finds the one record matching a field-level prompt's scope."""
    matches = du.filter_records(records, uid=scope.get("uid"), commodity_category=scope.get("commodity_category"),
                                 year=scope.get("year"), month=scope.get("month"))
    return matches[0] if matches else None


def fmt(v):
    """Formats a raw field value for display text: bools as yes/no,
    floats without trailing zeros, everything else as-is."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def summarize_tool_out(tool_out, max_items=5):
    """Trims a long tool-output list down for display in the prompt's
    context text. This only affects what appears in the prompt; the
    full tool_out is still used everywhere scoring happens."""
    if isinstance(tool_out, list):
        if len(tool_out) <= max_items:
            return tool_out
        return tool_out[:max_items] + [f"... and {len(tool_out) - max_items} more"]
    return tool_out


def gen_pair(p, records, rng):
    """Builds one full training pair from a single prompt: the real
    grounded answer, a deliberately broken one, and the prompt text
    that would be sent to a model."""
    if p["family"] == "tool":
        good, tool_out = A.answer(records, p["intent"], p["scope"])
        # computed before break_it() so break_it() can use it to decide
        # whether an unwarranted-hedge flaw applies - see break_it()'s
        # docstring. wrapped in bool() since tool_out.get("uncertain")
        # can be None (key absent) rather than a real False.
        uncertain = bool(isinstance(tool_out, dict) and (tool_out.get("status") == "insufficient_data" or tool_out.get("uncertain")))
        bad = A.break_it(good, p["scope"], rng, uncertain=uncertain)
        context = f"tool call: {p['intent']}({p['scope']}) -> {summarize_tool_out(tool_out)}"
    else:
        rec = find_record(records, p["scope"])
        if rec is None:
            good = "FACTS: no matching record\nRECOMMENDATION: can't say without data"
            bad = "FACTS: on record, everything's at 55%\nRECOMMENDATION: looks fine"
            tool_out = None
            uncertain = True
            context = "no record found"
        else:
            facts = "; ".join(f"{f}={fmt(rec[f])}" for f in p["fields"] if f in rec)
            good = f"FACTS: {facts}\nRECOMMENDATION: looks ok" if "not adequate" not in facts else f"FACTS: {facts}\nRECOMMENDATION: worth checking"
            mode = rng.choice(["wrong_val", "wrong_wh", "no_answer"])
            if mode == "wrong_val":
                m = re.search(r"\d+\.?\d*", good)
                bad = good[:m.start()] + str(round(float(m.group()) + rng.uniform(10, 30), 1)) + good[m.end():] if m else good + " (approx)"
            elif mode == "wrong_wh":
                bad = good.replace(rec["warehouse_id"], f"WH_{rng.randint(100,899):04d}")
            else:
                bad = "not sure, check the system"
            tool_out = {f: rec[f] for f in p["fields"] if f in rec}
            uncertain = False
            context = "record: " + facts

    prompt_text = f"{SYS}\n\n{context}\n\nQ: {p['text']}\nA:"
    return {
        "qid": p["qid"], "prompt": prompt_text, "good": good, "bad": bad,
        "tool_out": tool_out, "scope": p["scope"], "uncertain": uncertain,
        "family": p["family"], "intent": p["intent"],
    }


def gen_all(prompts, records, seed=42):
    """Generates a training pair for every prompt in the list, all
    drawing from one shared, seeded random generator for reproducible
    output."""
    rng = random.Random(seed)
    return [gen_pair(p, records, rng) for p in prompts]


def score(text, tool_out, scope, uncertain, valid_categories=None, valid_reasons=None,
          valid_levels=None, valid_regions=None):
    """Scores one response text against 7 rubric dimensions, using the
    real tool_out as ground truth. Returns (total, dims), where total is
    an equal-weighted average across all 7 dimensions and dims is the
    per-dimension breakdown. Used both to label chosen/rejected during
    training-pair generation and to score a model's live responses
    during evaluation."""
    dims = {}
    tl = text.lower()

    # fields a well-formed answer legitimately never restates literally,
    # either because they're redundant with something else that IS
    # checked ('uid' with warehouse_id+level) or represented indirectly
    # rather than as a literal word ('status' by which response branch
    # ran at all; 'has_issue'/'uncertain' by whether reasons/notes get
    # included, not by the literal word "True")
    NON_NARRATIVE_FIELDS = {"uid", "status", "has_issue", "uncertain"}

    # a compare-type answer's tool_out has a different shape entirely:
    # {"_grouped_by": dim, "groups": {group_val: <normal tool_out per group>}}
    # - detected up front so grounding and hallucination checks build
    # their "truth" from every group's data, not just tool_out's own
    # (nonexistent) top-level fields
    is_grouped = isinstance(tool_out, dict) and "_grouped_by" in tool_out

    true_vals = []
    true_whs = set()
    if "uid" in scope:
        # the scope's own uid is always a legitimate real warehouse to
        # mention, regardless of what tool_out looks like - matters for
        # an empty-result answer (e.g. "no active stockouts"), where
        # tool_out alone wouldn't otherwise establish which warehouse is
        # correct to name
        true_whs.add(scope["uid"].split(":", 1)[-1])
    if is_grouped:
        # whichever field IS the group_by axis is redundant with the
        # group label already checked separately, and the OTHER
        # cross-cutting dimension isn't relevant either - a "compare by
        # region" answer isn't expected to also report warehouse_level
        grouped_exclude = NON_NARRATIVE_FIELDS | {"warehouse_level", "region_type"}
        for group_val, group_res in tool_out["groups"].items():
            true_vals.append(("_group_label", group_val))
            top = None
            if isinstance(group_res, list) and group_res:
                top = group_res[0]
            elif isinstance(group_res, dict):
                top = group_res
            if top:
                for k, v in top.items():
                    if k not in grouped_exclude and isinstance(v, (int, float, str, bool)):
                        true_vals.append((k, v))
                if "warehouse_id" in top:
                    true_whs.add(top["warehouse_id"])
    elif isinstance(tool_out, dict):
        true_vals = [(k, v) for k, v in tool_out.items()
                     if k not in NON_NARRATIVE_FIELDS and isinstance(v, (int, float, str, bool))]
        if "warehouse_id" in tool_out:
            true_whs.add(tool_out["warehouse_id"])
    elif isinstance(tool_out, list) and tool_out:
        true_vals = [(k, v) for k, v in tool_out[0].items()
                     if k not in NON_NARRATIVE_FIELDS and isinstance(v, (int, float, str, bool))]
        true_vals.append(("_count", len(tool_out)))
        if "warehouse_id" in tool_out[0]:
            true_whs.add(tool_out[0]["warehouse_id"])

    if not true_vals and "uid" in scope:
        # an empty tool_out (e.g. a correct "no active stockouts for
        # this warehouse" answer) has no fields to check content
        # against - the one thing such an answer should still get right
        # is which warehouse it's talking about, so that becomes the
        # required fact here
        true_vals = [("_scope_subject", scope["uid"].split(":", 1)[-1])]

    if not true_vals:
        dims["grounding"] = 1.0
    else:
        # a stray space inside a decimal number ("45. 4%" instead of
        # "45.4%") would otherwise parse as two unrelated numbers rather
        # than one correctly-stated value, so that's normalized first
        text_for_numbers = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)
        stated_numbers = set(re.findall(r"-?\d+\.?\d*", text_for_numbers))
        # numeric tolerance: "82.0" (the raw stored value) and "82" (how
        # a model naturally drops a trailing zero) are the same number
        stated_floats = []
        for tok in stated_numbers:
            try:
                stated_floats.append(float(tok))
            except ValueError:
                pass
        # same idea for strings: "national_CMS" and "National CMS" are
        # the same tier, not a wrong one - normalize case and
        # underscore-vs-space before comparing
        text_norm = tl.replace("_", " ")

        # group labels are checked separately below, not through this
        # generic v_norm/text_norm path - see the loop
        facts_only = text.split("\nRECOMMENDATION:")[0]

        hits = 0
        for k, v in true_vals:
            if isinstance(v, bool):
                # booleans get phrased as "yes"/"no" elsewhere in this
                # codebase (see fmt()) - check both that and the raw
                # Python str() form defensively
                hits += ("yes" if v else "no") in tl or str(v) in text
            elif isinstance(v, (int, float)):
                hits += any(abs(f - float(v)) < 0.05 for f in stated_floats)
            elif k == "_group_label":
                # group labels need a real word-boundary match, not a
                # bare substring, and must appear within the FACTS
                # section specifically, not just anywhere in the text -
                # otherwise a group label mentioned only in the
                # recommendation, or one that's a substring of another
                # real label ("urban" inside "peri_urban"), could be
                # mistaken for a correct match. checking both underscore
                # and space phrasing (without normalizing the underscore
                # away first) is what makes the word-boundary check
                # work: regex \b treats "_" as a word character, so
                # \bperi_urban\b is one atomic token that \burban\b
                # can't match inside.
                raw_form = str(v)
                space_form = raw_form.replace("_", " ")
                pattern = r"(?i)\b(" + re.escape(raw_form) + "|" + re.escape(space_form) + r")\b"
                hits += re.search(pattern, facts_only) is not None
            else:
                v_norm = str(v).lower().replace("_", " ")
                hits += v_norm in text_norm
        dims["grounding"] = hits / len(true_vals)

    # tool/function selection accuracy is not scored here - it's
    # measured separately in eval_model.py's evaluate_tool_selection(),
    # which checks assistant.route() directly against the question text
    wh_mentioned = set(re.findall(r"WH_\d{4}", text, re.IGNORECASE))

    # operational_quality measures form/structure, independent of
    # whether the content is correct (grounding/no_hallucination's job)
    # or appropriately confident (uncertainty's job). three concrete,
    # checkable failure modes: structure, redundancy, completion.
    operational_quality_penalties = 0

    # 1. structural well-formedness: a well-formed answer states its
    # facts once and its recommendation once. repeating "FACTS:"/
    # "Fact:"/"RECOMMENDATION:" labels beyond that means the response is
    # re-starting itself instead of finishing.
    label_repeats = len(re.findall(r"\bfacts?:|\brecommendation:", tl))
    if label_repeats > 2:  # one FACTS:/Fact: + one RECOMMENDATION: is normal
        operational_quality_penalties += 1

    # 2. non-redundancy: does the response restate the same fact 3+
    # times? grounding only checks a true value appears at least once,
    # so a response stating the same number three times across three
    # sentences would otherwise still score grounding=1.0. threshold is
    # 3, not 2, since stating a number once in an intro sentence and
    # once more in a "Fact:" line is normal, non-degenerate style, not a
    # quality problem. warehouse ids are stripped first so a warehouse
    # mentioned twice is never counted as a repeated fact.
    #
    # two sources need excluding to avoid false positives on genuinely
    # good responses: an issue_counts dict literal has distinct keys
    # that can coincidentally share the same count (11 different real
    # facts, not one fact restated), so dict-literal substrings are
    # stripped before counting; and a compare-mode answer can have
    # multiple distinct groups genuinely tie on the same real number,
    # which is a real coincidence in the data, not redundant phrasing -
    # so this check is skipped entirely for grouped/compare-type
    # answers (the purpose-built compare-mode flaw check already covers
    # the real failure this is meant to catch there: a pooled figure
    # copied across groups instead of each group's own distinct number).
    if is_grouped:
        pass  # skipped entirely for grouped/compare-type answers
    else:
        text_no_ids = re.sub(r"WH_\d+", "", text, flags=re.IGNORECASE)
        text_no_dicts = re.sub(r"\{[^}]*\}", "", text_no_ids)
        text_for_nums = re.sub(r"(\d)\.\s+(\d)", r"\1\2", text_no_dicts)  # same decimal-space fix as grounding
        num_counts = {}
        for n in re.findall(r"-?\d+\.?\d*", text_for_nums):
            if len(n.replace(".", "").lstrip("-")) >= 2:  # skip trivial single-digit numbers (e.g. "0")
                num_counts[n] = num_counts.get(n, 0) + 1
        if any(c >= 3 for c in num_counts.values()):
            operational_quality_penalties += 1

    # 3. completion: does the response end on a finished thought
    # (terminal punctuation), not cut off mid-word or mid-sentence?
    if not text.strip().endswith((".", "!", "?", ":", "\"")):
        operational_quality_penalties += 1

    dims["operational_quality"] = max(0.0, 1.0 - operational_quality_penalties / 3)

    # uncertainty handling: does the response hedge when the data
    # genuinely is uncertain/provisional, and NOT hedge when the data is
    # solid? note the name is about calibration, not raw hedge presence
    # - a confident, hedge-free answer on a fully-answerable question
    # scores 1.0 here, and a hedge on that same answerable question
    # scores 0.0, since hedging isn't rewarded for its own sake.
    says_unsure = ("not sure" in tl or "no matching record" in tl or "provisional" in tl
                   or "no record found" in tl or "check the system" in tl or "check yourself" in tl)
    if uncertain:
        dims["uncertainty"] = 1.0 if says_unsure else 0.0
    else:
        dims["uncertainty"] = 0.0 if says_unsure else 1.0

    # hallucination: an invented warehouse id not in the true output.
    # for a compare answer this checks against the set of every group's
    # real top warehouse id, not a single value.
    invented = wh_mentioned - true_whs
    dims["no_hallucination"] = 0.0 if invented else 1.0

    # fabricated commodity category / issue reason code: a string that
    # isn't just wrong (some other real value) but doesn't exist in the
    # dataset's schema at all. structural extraction, not semantic -
    # relies on the exact phrasing answer() always uses
    # ("(LEVEL, CATEGORY)" for stockouts/backorders/shortage_risk,
    # "reported issue = X" for explain_issue). if the response doesn't
    # use that phrasing at all, this can't tell either way, so it isn't
    # penalized - it only catches fabrication when a claim IS being made
    # in the expected shape.
    if valid_categories:
        cat_match = re.search(r"\(([a-zA-Z_]+),\s*([a-zA-Z_]+)\)", text)
        claimed_cat = cat_match.group(2) if cat_match else None
        dims["no_invented_category"] = (
            0.0 if claimed_cat and claimed_cat not in valid_categories else 1.0
        )
    else:
        dims["no_invented_category"] = 1.0

    if valid_reasons:
        # matches both explain_issue phrasings: the single-warehouse
        # "reported issue = X" and the tier/region/compare aggregate's
        # "most common issue...: X ("
        reason_match = re.search(r"reported issue = ([a-zA-Z_]+)", text)
        if not reason_match:
            reason_match = re.search(r"most common issue[^:]*:\s*([a-zA-Z_]+)", text)
        claimed_reason = reason_match.group(1) if reason_match else None
        dims["no_invented_reason"] = (
            0.0 if claimed_reason and claimed_reason not in valid_reasons else 1.0
        )
    else:
        dims["no_invented_reason"] = 1.0

    # same idea as no_invented_category/reason, for the two group-by
    # dimensions - a compare-type answer's group labels ("district_store",
    # "urban", etc) are a small closed real set with the same
    # fabrication risk. reuses assistant.py's own compare-facts parser
    # rather than duplicating the FACTS-line regex a second time.
    if valid_levels or valid_regions:
        parsed = A._parse_compare_facts(text)
        if parsed:
            _, dim, segments = parsed
            valid_set = valid_levels if dim == "warehouse_level" else (valid_regions if dim == "region_type" else None)
            if valid_set:
                invented_labels = [label for label, _ in segments if label not in valid_set]
                dims["no_invented_group"] = 0.0 if invented_labels else 1.0
            else:
                dims["no_invented_group"] = 1.0  # dim wasn't level or region (e.g. the "warehouse" fake-dim mode)
        else:
            dims["no_invented_group"] = 1.0
    else:
        dims["no_invented_group"] = 1.0

    total = sum(dims.values()) / len(dims)
    return total, dims


def label_pairs(pairs, valid_categories=None, valid_reasons=None, valid_levels=None, valid_regions=None):
    """Scores both candidates in every pair and assigns chosen/rejected
    by whichever scores higher. Keeps tool_out/scope/uncertain/family/
    intent alongside chosen/rejected, since eval_model.py needs these to
    score a model's own generated text against ground truth."""
    out = []
    for p in pairs:
        good_score, _ = score(p["good"], p["tool_out"], p["scope"], p["uncertain"],
                               valid_categories, valid_reasons, valid_levels, valid_regions)
        bad_score, _ = score(p["bad"], p["tool_out"], p["scope"], p["uncertain"],
                              valid_categories, valid_reasons, valid_levels, valid_regions)
        if good_score >= bad_score:
            chosen, rejected = p["good"], p["bad"]
        else:
            chosen, rejected = p["bad"], p["good"]
        out.append({
            "qid": p["qid"], "prompt": p["prompt"], "chosen": chosen, "rejected": rejected,
            "chosen_score": max(good_score, bad_score), "rejected_score": min(good_score, bad_score),
            "tool_out": p["tool_out"], "scope": p["scope"], "uncertain": p["uncertain"],
            "family": p["family"], "intent": p["intent"],
        })
    return out


if __name__ == "__main__":
    recs = du.load_data(use_sample=True)
    prompts = build_prompts(recs, min_prompts=100)
    print(len(prompts), "prompts")
    pairs = gen_all(prompts, recs)
    valid_cats, valid_reasons = du.valid_categories_and_reasons(recs)
    valid_levels, valid_regions = du.valid_levels_and_regions(recs)
    labeled = label_pairs(pairs, valid_cats, valid_reasons, valid_levels, valid_regions)
    print(labeled[0])
    correct = sum(1 for p, l in zip(pairs, labeled) if l["chosen"] == p["good"])
    print(f"{correct}/{len(pairs)} chose the good one")
