"""Loads the warehouse inventory dataset and provides the tool functions
the assistant calls to answer questions.

The dataset (electricsheepafrica/warehouse-inventory-management) ships
as 3 CSV files, one per warehouse tier: district_store, national_CMS,
regional_warehouse.

Gotcha: warehouse_id ("WH_0001".."WH_0099") is only unique WITHIN one of
those 3 files, not across them - all 3 files reuse the same 99 ids for
different physical warehouses. A "uid" field (warehouse_level +
warehouse_id) is added to every record as the real unique key, and is
used everywhere a single warehouse needs to be pinned down unambiguously.

inventory_issue is a real reason code ("theft_pilferage", "expired_stock",
"none", etc - 12 values total), not a yes/no flag, so
explain_inventory_issue reports the actual recorded reason as its
primary fact rather than inferring one.
"""

import csv
import json
import os

FLAG_FIELDS = {
    "has_WMS", "has_temperature_monitoring", "has_generator_backup", "has_pest_control",
    "has_security_system", "storage_conditions_adequate", "stock_record_up_to_date",
    "fefo_compliance", "overflow_to_other_space", "temperature_excursion_month",
    "pest_damage_reported", "theft_reported", "stockout_at_warehouse", "report_submitted",
}
INT_FIELDS = {
    "id", "warehouse_size_sqm", "staff_count", "staff_trained_gdp", "year", "month",
    "orders_received_month", "orders_fulfilled_complete", "orders_backordered",
    "facilities_affected_by_stockout",
}
FLOAT_FIELDS = {
    "volume_share_pct", "inventory_accuracy_pct", "order_fulfilment_rate_pct",
    "wastage_rate_pct", "expired_stock_value_usd", "damaged_goods_value_usd",
    "capacity_utilisation_pct",
}

CSV_FILES = {
    "district_store": "warehouse_district_store.csv",
    "national_CMS": "warehouse_national_central_medical_store.csv",
    "regional_warehouse": "warehouse_regional_warehouse.csv",
}

# critical is the top severity tier in this dataset
_CRITICALITY_WEIGHT = {"critical": 2.0, "high": 1.5, "medium": 0.5, "low": 0.0}


def _cast_row(row):
    """Converts a raw CSV row's string values to their real types (bool,
    int, float) and adds the uid field."""
    for f in FLAG_FIELDS:
        row[f] = bool(int(row[f]))
    for f in INT_FIELDS:
        row[f] = int(row[f])
    for f in FLOAT_FIELDS:
        row[f] = float(row[f])
    row["uid"] = f"{row['warehouse_level']}:{row['warehouse_id']}"
    return row


def load_from_csv_dir(data_dir="data"):
    """Loads the 3 real source CSVs from a folder. Place the 3 files
    (same names as CSV_FILES values) in data_dir and this reads all of
    them into one flat list of records."""
    records = []
    for level, fname in CSV_FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"expected {path} - place the 3 source CSVs in {data_dir}/"
            )
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                records.append(_cast_row(row))
    return records


def load_data(use_sample=False, data_dir="data"):
    """Loads either the small stratified sample (warehouse_sample.jsonl)
    or the full 3-CSV dataset, depending on use_sample."""
    if use_sample:
        return [json.loads(l) for l in open(f"{data_dir}/warehouse_sample.jsonl") if l.strip()]
    return load_from_csv_dir(data_dir)


def filter_records(records, uid=None, warehouse_id=None, commodity_category=None, year=None, month=None,
                    warehouse_level=None, region_type=None):
    """Narrows a record list down by any combination of the given
    fields. Any filter left as None is not applied."""
    out = records
    if uid:
        out = [r for r in out if r["uid"] == uid]
    if warehouse_id:
        out = [r for r in out if r["warehouse_id"] == warehouse_id]
    if commodity_category:
        out = [r for r in out if r["commodity_category"] == commodity_category]
    if year:
        out = [r for r in out if r["year"] == year]
    if month:
        out = [r for r in out if r["month"] == month]
    if warehouse_level:
        out = [r for r in out if r["warehouse_level"] == warehouse_level]
    if region_type:
        out = [r for r in out if r["region_type"] == region_type]
    return out


def valid_levels_and_regions(records):
    """Returns the real sets of warehouse_level and region_type values
    present in the data, for validating a claimed tier/region against."""
    levels = {r["warehouse_level"] for r in records}
    regions = {r["region_type"] for r in records}
    return levels, regions


def grouped(records, group_by, tool_fn, group_values=None, **tool_kwargs):
    """Runs tool_fn once per distinct value of group_by (warehouse_level
    or region_type), instead of once over all records. group_values, if
    given, restricts to just those group values (a named comparison like
    "district vs national" instead of comparing across every tier).
    tool_fn must accept a list of records as its first positional arg.

    The result is wrapped in a dict with a "_grouped_by" marker key so
    score() (gen_pairs.py) can tell this apart from a normal
    single-answer dict result without guessing from the keys alone.
    """
    buckets = {}
    for r in records:
        buckets.setdefault(r[group_by], []).append(r)
    if group_values:
        buckets = {k: v for k, v in buckets.items() if k in group_values}
    results = {g: tool_fn(recs_g, **tool_kwargs) for g, recs_g in buckets.items()}
    return {"_grouped_by": group_by, "groups": results}


def valid_categories_and_reasons(records):
    """Returns the real sets of commodity_category and inventory_issue
    values present in the data, derived from the data itself so this
    can't go stale if the dataset's category/reason list changes."""
    categories = {r["commodity_category"] for r in records}
    reasons = {r["inventory_issue"] for r in records}
    return categories, reasons


def find_uids_for_bare_id(records, warehouse_id):
    """A bare id like "WH_0009" can match up to 3 different real
    warehouses (one per tier). Returns the list of (uid, warehouse_level)
    it actually matches."""
    seen = {}
    for r in records:
        if r["warehouse_id"] == warehouse_id:
            seen[r["uid"]] = r["warehouse_level"]
    return sorted(seen.items())


def get_stockouts(records, **filters):
    """Returns every record with an active stockout matching the given
    filters, sorted by how many facilities are affected (worst first)."""
    rows = filter_records(records, **filters)
    out = [r for r in rows if r.get("stockout_at_warehouse")]
    out.sort(key=lambda r: r.get("facilities_affected_by_stockout", 0), reverse=True)
    return [{
        "uid": r["uid"], "warehouse_id": r["warehouse_id"], "warehouse_level": r["warehouse_level"],
        "commodity_category": r["commodity_category"],
        "facilities_affected_by_stockout": r["facilities_affected_by_stockout"],
    } for r in out]


def get_backorders(records, **filters):
    """Returns every record with a nonzero backorder count matching the
    given filters, sorted by backorder rate (orders backordered ÷ orders
    received) rather than the raw count, since a raw count is misleading
    across warehouses of very different scale."""
    rows = filter_records(records, **filters)
    out = []
    for r in rows:
        if r.get("orders_backordered", 0) > 0:
            recv = r.get("orders_received_month", 0)
            rate = round(100 * r["orders_backordered"] / recv, 1) if recv else 0
            out.append({
                "uid": r["uid"], "warehouse_id": r["warehouse_id"], "warehouse_level": r["warehouse_level"],
                "commodity_category": r["commodity_category"],
                "orders_backordered": r["orders_backordered"], "backorder_rate_pct": rate,
            })
    out.sort(key=lambda x: x["backorder_rate_pct"], reverse=True)
    return out


def rank_shortage_risk(records, top_n=10, **filters):
    """Scores every matching record on shortage risk, using a
    transparent, hand-weighted heuristic (not a fitted model - there is
    no ground-truth "did a shortage actually occur" label to calibrate
    against). Returns the top_n highest-scoring records along with the
    specific factors that contributed to each score."""
    records = filter_records(records, **filters)
    scored = []
    for r in records:
        recv = r.get("orders_received_month", 0)
        backorder_rate = 100 * r["orders_backordered"] / recv if recv else 0
        score = 0
        factors = []
        if r.get("stockout_at_warehouse"):
            score += 3
            factors.append("stockout")
        if backorder_rate > 25:
            score += 2
            factors.append("high_backorder_rate")
        if r.get("order_fulfilment_rate_pct", 100) < 75:
            score += 2
            factors.append("low_fulfilment")
        if r.get("wastage_rate_pct", 0) > 5:
            score += 1
            factors.append("high_wastage")
        if not r.get("storage_conditions_adequate", True):
            score += 1.5
            factors.append("storage_inadequate")
        crit_w = _CRITICALITY_WEIGHT.get(r.get("criticality"), 0)
        if crit_w:
            score += crit_w
            factors.append(f"criticality_{r['criticality']}")
        if score > 0:
            scored.append({
                "uid": r["uid"], "warehouse_id": r["warehouse_id"], "warehouse_level": r["warehouse_level"],
                "commodity_category": r["commodity_category"],
                "risk_score": score, "risk_factors": factors,
            })
    scored.sort(key=lambda x: x["risk_score"], reverse=True)
    return scored[:top_n]


def _compute_kpis(rows):
    """Aggregates KPI stats over whatever record list it's given. Kept
    separate from get_warehouse_kpis so the same averaging logic can run
    once per warehouse, or once per tier/region group via grouped(),
    without duplicating the math."""
    if not rows:
        return None
    avg = lambda field: round(sum(r[field] for r in rows) / len(rows), 1)
    issue_counts = {}
    for r in rows:
        if r["inventory_issue"] != "none":
            issue_counts[r["inventory_issue"]] = issue_counts.get(r["inventory_issue"], 0) + 1
    return {
        "n_records": len(rows),
        "avg_inventory_accuracy_pct": avg("inventory_accuracy_pct"),
        "avg_order_fulfilment_rate_pct": avg("order_fulfilment_rate_pct"),
        "months_with_stockout": sum(1 for r in rows if r.get("stockout_at_warehouse")),
        "issue_counts": issue_counts,
    }


def get_kpis(records, **filters):
    """Generalizes get_warehouse_kpis to any filter (warehouse_level,
    region_type, uid, or none at all for a fully dataset-wide aggregate).
    get_warehouse_kpis stays as its own function below rather than
    calling this one, since its return shape adds uid/warehouse_id/
    warehouse_level fields that only make sense for exactly one
    warehouse - meaningless for a multi-warehouse aggregate."""
    rows = filter_records(records, **filters)
    return _compute_kpis(rows)


def get_warehouse_kpis(records, uid):
    """Aggregates one specific warehouse's records into summary KPIs:
    average inventory accuracy, average fulfilment rate, how many months
    had a stockout, and a breakdown of recurring issue types."""
    rows = [r for r in records if r["uid"] == uid]
    if not rows:
        return None
    result = _compute_kpis(rows)
    result.update({"uid": uid, "warehouse_id": rows[0]["warehouse_id"], "warehouse_level": rows[0]["warehouse_level"]})
    return result


def get_dominant_issue(records, **filters):
    """Returns the most common real inventory_issue value among matching
    records (excluding "none"), grounded in an actual count. Used for
    tier/region-scoped "why" questions, where - unlike a single
    warehouse - there is no one real record to point to as the reason:
    different warehouses in the same tier/region can fail for different
    real reasons, so the most common one is the only grounded aggregate
    answer that doesn't fabricate a summary explanation no single record
    actually supports."""
    rows = filter_records(records, **filters)
    counts = {}
    for r in rows:
        if r["inventory_issue"] != "none":
            counts[r["inventory_issue"]] = counts.get(r["inventory_issue"], 0) + 1
    if not counts:
        return {"status": "insufficient_data", "reason": "no recorded issues for that scope"}
    dominant_issue, count = max(counts.items(), key=lambda kv: kv[1])
    return {"dominant_issue": dominant_issue, "count": count, "n_records": len(rows), "issue_counts": counts}


def explain_inventory_issue(records, uid, commodity_category, year=None, month=None):
    """Looks up the real recorded reason for an issue at one warehouse
    and commodity category. If no exact year/month is given, falls back
    to the most recent matching record and discloses that explicitly.
    Also flags the result as uncertain if the underlying report was
    never submitted, since its other figures are then provisional."""
    assumed_period = False

    if year is not None and month is not None:
        matches = filter_records(records, uid=uid, commodity_category=commodity_category, year=year, month=month)
    else:
        candidates = filter_records(records, uid=uid, commodity_category=commodity_category)
        if candidates:
            candidates.sort(key=lambda r: (r["year"], r["month"]), reverse=True)
            matches = [candidates[0]]
            assumed_period = True
        else:
            matches = []

    if not matches:
        return {"status": "insufficient_data", "reason": "no record found for that warehouse/period"}

    r = matches[0]
    # the real recorded reason - a grounded fact, not an inference
    reported_issue = r["inventory_issue"]

    # additional context signals, kept separate from the reported reason
    # rather than presented as if they were themselves the cause
    signals = []
    if not r.get("storage_conditions_adequate", True):
        signals.append("storage conditions not adequate")
    if not r.get("fefo_compliance", True):
        signals.append("FEFO compliance failed")
    if r.get("inventory_accuracy_pct", 100) < 85:
        signals.append(f"low inventory accuracy ({r['inventory_accuracy_pct']}%)")
    if r.get("wastage_rate_pct", 0) > 5:
        signals.append(f"high wastage ({r['wastage_rate_pct']}%)")
    if r.get("stockout_at_warehouse"):
        signals.append("active stockout")

    result = {
        "status": "ok",
        "uid": uid, "warehouse_id": r["warehouse_id"], "warehouse_level": r["warehouse_level"],
        "year": r["year"], "month": r["month"],
        "reported_issue": reported_issue,
        "has_issue": reported_issue != "none",
        "signals": signals,
    }
    if assumed_period:
        result["uncertain"] = True
        result["note"] = f"no period given, used most recent record ({r['month']}/{r['year']})"
    if not r.get("report_submitted", True):
        result["uncertain"] = True
    return result


if __name__ == "__main__":
    recs = load_data(use_sample=True)
    print(len(recs), "records loaded")
    print("stockouts:", get_stockouts(recs)[:2])
    print("risk:", rank_shortage_risk(recs, top_n=3))
    wh = recs[0]["uid"]
    print("kpis:", get_warehouse_kpis(recs, wh))
