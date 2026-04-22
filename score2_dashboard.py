import json
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st_ui

from ETF.core.constants import (
    DASHBOARD_CACHE_BUNDLE_VERSION,
    DASHBOARD_CACHE_TTL_SECONDS,
    DASHBOARD_HODL_EXCHANGE_URL,
    DASHBOARD_HODL_LOGO_URL,
    DASHBOARD_PREFETCH_INTERVAL_SECONDS,
    DASHBOARD_TITLE,
    IMM_DATA,
    IMM_RATIO,
    IMM_WINDOW,
    NETUID,
    EPOCHES_IN_DAY,
)
from ETF.core.functions import (
    _get_all_subnets_cached,
    _get_metagraph_cached,
    _get_stake_info_for_coldkeys_cached,
    score1,
)

# Server-wide cache: same Streamlit process shares entries for all browser sessions.
# Multiple app replicas (e.g. K8s) each have their own cache unless you add Redis/DB.
_CACHE_TTL_SECONDS = DASHBOARD_CACHE_TTL_SECONDS
_PREFETCH_INTERVAL_SECONDS = DASHBOARD_PREFETCH_INTERVAL_SECONDS
# Bump when the cached dict shape changes (invalidates stale on-disk cache entries).
_CACHE_BUNDLE_VERSION = DASHBOARD_CACHE_BUNDLE_VERSION

HODL_LOGO_URL = DASHBOARD_HODL_LOGO_URL
HODL_EXCHANGE_URL = DASHBOARD_HODL_EXCHANGE_URL

_TOOLTIP_ALLOC = {
    "rank": "Global rank among all miners by final blended score (metagraph score with simple score overriding where IMM applies). Lower number is higher rank.",
    "uid": "Miner UID on subnet.",
    "hotkey": "Miner hotkey (SS58).",
    "coldkey": "Coldkey (SS58) holding stake.",
    "count": "Number of UIDs on this coldkey; divides per-UID score in validator logic.",
    "daily_reward": "Current miner daily reward from subnet metagraph (emission * EPOCHES_IN_DAY) in selected currency.",
    "total (score)": "IMM wallet total volume fed into simple score (type-1 + capped type-2 from IMM window); this is the score basis before ratio scaling and per-coldkey split.",
    "score": "Simple score incentive weight for this miner after normalizing wallet totals (matches validator simple score path).",
}
_TOOLTIP_IMM_WALLET_ROW = {
    "wallet": "Coldkey (SS58) from IMM.",
    "asset": "Subnet netuid IMM attributes this row to.",
    "volume": "Summed IMM volume for this wallet/asset/type (TAO-equivalent per IMM feed).",
}
_TOOLTIP_WALLET_TOTALS = {
    "wallet": "Coldkey (SS58).",
    "total": "Per-wallet IMM total (type-1 + min(type-2 volume, on-chain stake cap)); used as simple score numerator input.",
}
_TOOLTIP_INCENTIVE_RATIO = (
    "How to read this: [burn, ETF, IMM]."
    "Example: [0.8, 0.0, 0.2] means 80% burn, 0% ETF miners (deprecated), and 20% IMM miners."
)
_TOOLTIP_IMM_WINDOW = (
    "Number of past days included in IMM data. "
    "Example: 7 means volumes from the last 7 days are used."
)
_TOOLTIP_IMM_DATE_KEY = (
    "IMM snapshot key generated from the window. "
    "It is the timestamp used to request IMM data for this run."
)


def _column_config_for_df(df: pd.DataFrame, help_by_col: dict[str, str]) -> dict:
    """Streamlit column_config with a help tooltip on every column."""
    cc = st_ui.column_config
    out: dict = {}
    for col in df.columns:
        help_text = help_by_col.get(col) or f"Column `{col}`."
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            out[col] = cc.CheckboxColumn(help=help_text)
        elif pd.api.types.is_integer_dtype(s):
            out[col] = cc.NumberColumn(help=help_text, format="%d")
        elif pd.api.types.is_float_dtype(s):
            out[col] = cc.NumberColumn(help=help_text)
        else:
            out[col] = cc.TextColumn(help=help_text)
    return out


def dtms(days: int = 0) -> str:
    return (datetime.utcnow() + timedelta(days=days)).isoformat(" ", "milliseconds")


def _filter_alloc_rows_by_key(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Filter score rows by coldkey/hotkey substring (case-insensitive)."""
    q = (query or "").strip().lower()
    if not q:
        return df
    cold = df["coldkey"].astype(str).str.lower().str.contains(q, na=False) if "coldkey" in df.columns else False
    hot = df["hotkey"].astype(str).str.lower().str.contains(q, na=False) if "hotkey" in df.columns else False
    mask = cold | hot
    return df[mask].copy()


def _load_imm_inputs():
    ratio = json.loads(requests.get(IMM_RATIO, timeout=30).json())
    window = json.loads(requests.get(IMM_WINDOW, timeout=30).json())
    date = dtms(days=-window)
    base = pd.DataFrame(json.loads(requests.get(f"{IMM_DATA}/{date}", timeout=30).json()))
    cols = ["wallet", "asset", "type", "tao"]
    base = base[cols].copy()
    base.columns = [*base.columns[:-1], "volume"]
    return ratio, window, date, base


@st_ui.cache_data(ttl=_CACHE_TTL_SECONDS)
def _get_tao_usd_price() -> float | None:
    try:
        data = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd",
            timeout=10,
        ).json()
        return float(data["bittensor"]["usd"])
    except Exception:
        return None


def compute_score2_breakdown(sc: pd.DataFrame):
    jj = ["wallet", "asset", "type", "volume"]
    ratio, window, date, base = _load_imm_inputs()

    d1 = base[base["type"] == 1].groupby(jj[:-1]).sum().reset_index()
    d2 = base[base["type"] == 2].groupby(jj[:-1]).sum().reset_index()

    nn = _get_all_subnets_cached()
    kk = sc[sc["index"].isna()]["coldkey"].unique()
    kk = [ck for ck in kk if ck in d2["wallet"].unique()]
    d3 = pd.DataFrame(columns=jj)
    stake_info_by_ck = _get_stake_info_for_coldkeys_cached(kk)
    for ck in kk:
        sn = d2[d2["wallet"] == ck]["asset"].unique()
        for stake_info in stake_info_by_ck.get(ck, []):
            if stake_info.netuid not in sn:
                continue
            d3.loc[len(d3)] = (
                ck,
                stake_info.netuid,
                2,
                float(stake_info.stake) * float(nn[stake_info.netuid].price),
            )
    d3 = d3.groupby(jj[:-1]).sum().reset_index() if len(d3) else d3

    # Join miner volume (d2) to current stake snapshot (d3), then cap by min(volume, stake).
    d2_stake = d2.join(d3.set_index(jj[:-1])["volume"], jj[:-1], rsuffix="_stake")
    d2_stake.columns = [*d2_stake.columns[:-1], "stake"]
    d2_stake.loc[d2_stake["stake"].isna(), "stake"] = 0
    d2_capped = d2_stake.copy()
    d2_capped["volume"] = d2_capped[["volume", "stake"]].min(1)
    d2_capped = d2_capped.drop("stake", axis=1)

    merged = pd.concat([d1, d2_capped])
    totals = merged.groupby(jj[:1]).sum().reset_index().drop(jj[1:-1], axis=1)
    totals.columns = [*totals.columns[:-1], "total"]

    scored = sc.join(totals.set_index(jj[:1])["total"], "coldkey")
    scored = scored.drop(sc.columns[-4:], axis=1).dropna(subset="total")

    # Keep this aligned with ETF.core.functions.score2 semantics:
    # ratio = [brn_weight, score1_weight, score2_weight] (new),
    # with backward support for legacy 2-value payloads.
    score1_weight = ratio[1] if len(ratio) > 1 else 0.0
    score2_weight = ratio[2] if len(ratio) > 2 else (ratio[1] if len(ratio) > 1 else 0.0)
    scored["score2"] = scored["total"]
    if not score2_weight:
        scored["score2"] = 0.0
    elif score1_weight:
        dfz = sc["score"].sum() * score2_weight / score1_weight
        if dfz and scored["score2"].sum():
            scored["score2"] = dfz * scored["score2"] / scored["score2"].sum()
    scored["score2"] /= scored["count"]

    # Full metagraph final score (same merge as ETF.core.functions.score2).
    metagraph_final = sc.merge(scored[["uid", "score2"]], on="uid", how="left")
    metagraph_final.loc[metagraph_final["score2"].notna(), "score"] = metagraph_final["score2"]
    metagraph_final = metagraph_final.drop(columns=["score2"])

    rank_by_uid = _rank_by_uid_from_metagraph(metagraph_final)
    mg = _get_metagraph_cached(NETUID)
    subnet_price = float(nn[NETUID].price)
    emission_by_uid = {uid: float(em) * EPOCHES_IN_DAY for uid, em in enumerate(mg.emission)}
    scored["daily_reward"] = scored["uid"].map(emission_by_uid)

    return {
        "ratio": ratio,
        "window": window,
        "date": date,
        "raw": base,
        "d2_type2_raw": d2,
        "d3_chain_stake": d3,
        "d2_type2_capped": d2_capped,
        "wallet_totals": totals,
        "scored_rows": scored,
        "subnet_price_tao": subnet_price,
        "metagraph_final": metagraph_final,
        "rank_by_uid": rank_by_uid,
    }


def _rank_by_uid_from_metagraph(metagraph_final: pd.DataFrame) -> pd.Series:
    ranked = metagraph_final.sort_values(
        ["score", "uid"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    return pd.Series(ranked.index + 1, index=ranked["uid"].values, name="rank")


@st_ui.cache_data(
    ttl=_CACHE_TTL_SECONDS,
    show_spinner="Fetching chain + IMM (then served from cache until TTL expires)…",
)
def load_score2_bundle(netuid: int, _cache_bundle_version: int = _CACHE_BUNDLE_VERSION) -> dict:
    """Heavy work: score1 metagraph + stake + IMM APIs. Cached globally per `netuid` on this server."""
    _ = _cache_bundle_version  # cache key only; bump _CACHE_BUNDLE_VERSION when bundle keys change
    sc = score1(netuid=netuid)
    bundle = compute_score2_breakdown(sc)
    bundle["synced_at_utc"] = datetime.now(timezone.utc)
    return bundle


@st_ui.cache_resource
def _start_score2_prefetch(netuid: int) -> threading.Thread:
    """Once per process: warm cache immediately, then refresh on a fixed interval.

    Note: Streamlit only executes this script when at least one client has connected
    (or you hit the app once, e.g. health-check curl). It does not run before the
    first script run in a stock `streamlit run` setup.
    """

    def worker() -> None:
        while True:
            try:
                load_score2_bundle.clear()
                load_score2_bundle(netuid, _CACHE_BUNDLE_VERSION)
            except Exception:
                traceback.print_exc()
            time.sleep(_PREFETCH_INTERVAL_SECONDS)

    t = threading.Thread(target=worker, name="score2-prefetch", daemon=True)
    t.start()
    return t


def render():
    st_ui.set_page_config(
        page_title=DASHBOARD_TITLE,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _start_score2_prefetch(NETUID)

    st_ui.markdown(
        f"""<div style="display:flex;align-items:center;gap:0.375rem;margin:0 0 0.25rem 0;">
    <img src="{HODL_LOGO_URL}" width="64" height="64" style="object-fit:contain;flex-shrink:0;" alt="" />
    <h1 style="margin:0;padding:0;line-height:1.15;font-size:calc(1.75rem + 0.5vw);font-weight:600;">{DASHBOARD_TITLE}</h1>
</div>""",
        unsafe_allow_html=True,
    )
    st_ui.markdown(
        f"[HODL Exchange]({HODL_EXCHANGE_URL})"
    )
    st_ui.caption(
        "IMM volume → capped type-2 → wallet totals → simple score splits; "
        f"subnet {NETUID} metagraph matches ETF/core/functions.py."
    )

    try:
        data = load_score2_bundle(NETUID, _CACHE_BUNDLE_VERSION)
    except Exception as exc:
        st_ui.error(f"Failed: {exc}")
        st_ui.code(traceback.format_exc())
        return

    synced = data.get("synced_at_utc")
    sync_lbl = synced.strftime("%Y-%m-%d %H:%M:%S UTC") if isinstance(synced, datetime) else "—"
    st_ui.caption(
        f"One server-wide cached snapshot (TTL {_CACHE_TTL_SECONDS}s, prefetch every "
        f"{_PREFETCH_INTERVAL_SECONDS}s); last sync {sync_lbl}."
    )

    c1, c2, c3 = st_ui.columns(3)
    c1.metric("IMM window (days)", data["window"], help=_TOOLTIP_IMM_WINDOW)
    c2.metric("Incentive ratio", str(data["ratio"]), help=_TOOLTIP_INCENTIVE_RATIO)
    c3.metric("IMM date key", data["date"], help=_TOOLTIP_IMM_DATE_KEY)

    key_filter = st_ui.text_input(
        "Filter all tables by coldkey or hotkey",
        value="",
        placeholder="Paste full or partial coldkey/hotkey",
    )
    alpha_symbol = str(_get_all_subnets_cached()[NETUID].symbol)
    currency_options = ["alpha", "tao", "dollar"]
    currency_labels = {"alpha": alpha_symbol, "tao": "τ", "dollar": "$"}
    if "daily_reward_currency" not in st_ui.session_state:
        st_ui.session_state["daily_reward_currency"] = "alpha"
    selected_currency = st_ui.sidebar.selectbox(
        "Daily reward currency",
        currency_options,
        index=currency_options.index(st_ui.session_state["daily_reward_currency"]),
        format_func=lambda x: currency_labels.get(x, x),
        key="daily_reward_currency",
    )

    rank_by_uid = data.get("rank_by_uid")
    if rank_by_uid is None:
        mf = data.get("metagraph_final")
        rank_by_uid = _rank_by_uid_from_metagraph(mf) if mf is not None else pd.Series(dtype="int64")

    filtered_scored_rows = _filter_alloc_rows_by_key(data["scored_rows"], key_filter).copy()
    filtered_coldkeys = set(filtered_scored_rows["coldkey"].astype(str).unique())
    tao_usd = _get_tao_usd_price()
    tao_multiplier = float(data.get("subnet_price_tao", 1.0))
    usd_multiplier = float(tao_usd) if tao_usd is not None else 1.0
    reward_prefix_by_currency = {
        "alpha": f"{alpha_symbol} ",
        "tao": "τ ",
        "dollar": "$ ",
    }
    reward_multiplier_by_currency = {
        "alpha": 1.0,
        "tao": tao_multiplier,
        "dollar": tao_multiplier * usd_multiplier,
    }
    reward_prefix = reward_prefix_by_currency.get(selected_currency, f"{alpha_symbol} ")
    filtered_scored_rows["daily_reward"] = (
        filtered_scored_rows["daily_reward"]
        * reward_multiplier_by_currency.get(selected_currency, 1.0)
    )
    filtered_scored_rows["daily_reward"] = (
        filtered_scored_rows["daily_reward"].round(2).map(lambda v: f"{reward_prefix}{v}")
    )

    alloc = (
        filtered_scored_rows
        .sort_values(["score2", "uid"], ascending=[False, True], na_position="last")
        .drop(columns=["score"], errors="ignore")
        .rename(columns={"score2": "score"})
        .rename(columns={"total": "total (score)"})
        .drop(columns=["index"], errors="ignore")
        .reset_index(drop=True)
    )
    alloc.insert(0, "rank", alloc["uid"].map(rank_by_uid))
    if key_filter.strip():
        st_ui.caption(f"Filter active: `{key_filter.strip()}` · matching rows: {len(alloc)}")
    st_ui.subheader("score allocations")
    st_ui.dataframe(
        alloc,
        use_container_width=True,
        hide_index=True,
        column_config=_column_config_for_df(alloc, _TOOLTIP_ALLOC),
    )

    with st_ui.expander("IMM pipeline (intermediate tables)", expanded=False):
        st_ui.markdown("Step 1 - IMM mined volume by wallet and subnet (raw, simple-score-eligible events)")
        d2r = data["d2_type2_raw"]
        if key_filter.strip() and "wallet" in d2r.columns:
            d2r = d2r[d2r["wallet"].astype(str).isin(filtered_coldkeys)]
        d2r = d2r.drop(columns=["type"], errors="ignore")
        st_ui.dataframe(
            d2r,
            use_container_width=True,
            hide_index=True,
            column_config=_column_config_for_df(d2r, _TOOLTIP_IMM_WALLET_ROW),
        )
        st_ui.markdown("Step 2 - On-chain stake snapshot per wallet/subnet used as the cap")
        d3 = data["d3_chain_stake"]
        if key_filter.strip() and "wallet" in d3.columns:
            d3 = d3[d3["wallet"].astype(str).isin(filtered_coldkeys)]
        d3 = d3.drop(columns=["type"], errors="ignore")
        st_ui.dataframe(
            d3,
            use_container_width=True,
            hide_index=True,
            column_config=_column_config_for_df(
                d3,
                {
                    **_TOOLTIP_IMM_WALLET_ROW,
                    "volume": "On-chain stake value (TAO) used to cap type-2 IMM volume for this wallet/netuid.",
                },
            ),
        )
        st_ui.markdown("Step 3 - Capped mined volume after applying min(raw volume, on-chain stake)")
        d2c = data["d2_type2_capped"]
        if key_filter.strip() and "wallet" in d2c.columns:
            d2c = d2c[d2c["wallet"].astype(str).isin(filtered_coldkeys)]
        d2c = d2c.drop(columns=["type"], errors="ignore")
        st_ui.dataframe(
            d2c,
            use_container_width=True,
            hide_index=True,
            column_config=_column_config_for_df(
                d2c,
                {
                    **_TOOLTIP_IMM_WALLET_ROW,
                    "volume": "min(IMM type-2 volume, chain stake) for this wallet/netuid.",
                },
            ),
        )
        st_ui.markdown("Step 4 - Per-wallet total score basis used for allocation normalization")
        wt = data["wallet_totals"]
        if key_filter.strip() and "wallet" in wt.columns:
            wt = wt[wt["wallet"].astype(str).isin(filtered_coldkeys)]
        st_ui.dataframe(
            wt,
            use_container_width=True,
            hide_index=True,
            column_config=_column_config_for_df(wt, _TOOLTIP_WALLET_TOTALS),
        )
        raw2 = data["raw"][data["raw"]["type"] == 2]
        if key_filter.strip() and "wallet" in raw2.columns:
            raw2 = raw2[raw2["wallet"].astype(str).isin(filtered_coldkeys)]
        raw2 = raw2.drop(columns=["type"], errors="ignore")
        st_ui.markdown("Reference - Raw IMM event rows behind this snapshot")
        st_ui.dataframe(
            raw2,
            use_container_width=True,
            hide_index=True,
            column_config=_column_config_for_df(raw2, _TOOLTIP_IMM_WALLET_ROW),
        )

    csv_bytes = alloc.to_csv(index=False).encode("utf-8")
    st_ui.download_button("Download score allocations (CSV)", csv_bytes, file_name="score_allocations.csv")


if __name__ == "__main__":
    render()
