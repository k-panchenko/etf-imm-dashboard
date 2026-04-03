import json
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st_ui

from ETF.core.constants import IMM_DATA, IMM_RATIO, IMM_WINDOW, NETUID
from ETF.core.functions import score1, st as subtensor

# Server-wide cache: same Streamlit process shares entries for all browser sessions.
# Multiple app replicas (e.g. K8s) each have their own cache unless you add Redis/DB.
_CACHE_TTL_SECONDS = 3600
_PREFETCH_INTERVAL_SECONDS = 3600
# Bump when the cached dict shape changes (invalidates stale on-disk cache entries).
_CACHE_BUNDLE_VERSION = 1

HODL_LOGO_URL = "https://subnet118.com/logo1.png"
HODL_EXCHANGE_URL = "https://hodl.subnet118.com"
HODL_ETF_DASHBOARD_URL = "https://subnet-118-dashboard.vercel.app/"
DASHBOARD_TITLE = "HODL IMM Miner Dashboard"

_TOOLTIP_ALLOC = {
    "rank": "Global rank among all miners by final blended score (metagraph score with score2 overriding where IMM applies). Lower number is higher rank.",
    "uid": "Miner UID on subnet.",
    "hotkey": "Miner hotkey (SS58).",
    "coldkey": "Coldkey (SS58) holding stake.",
    "count": "Number of UIDs on this coldkey; divides per-UID score in validator logic.",
    "total": "IMM wallet total volume fed into score2 (type-1 + capped type-2 from IMM window).",
    "score2": "IMM incentive weight for this miner after normalizing wallet totals (matches validator score2 path).",
}
_TOOLTIP_IMM_WALLET_ROW = {
    "wallet": "Coldkey (SS58) from IMM.",
    "asset": "Subnet netuid IMM attributes this row to.",
    "type": "IMM event type from API (1 and 2 are aggregated differently in the validator).",
    "volume": "Summed IMM volume for this wallet/asset/type (TAO-equivalent per IMM feed).",
}
_TOOLTIP_WALLET_TOTALS = {
    "wallet": "Coldkey (SS58).",
    "total": "Per-wallet IMM total (type-1 + min(type-2 volume, on-chain stake cap)); used as score2 numerator input.",
}


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


def _load_imm_inputs():
    ratio = json.loads(requests.get(IMM_RATIO, timeout=30).json())
    window = json.loads(requests.get(IMM_WINDOW, timeout=30).json())
    date = dtms(days=-window)
    base = pd.DataFrame(json.loads(requests.get(f"{IMM_DATA}/{date}", timeout=30).json()))
    cols = ["wallet", "asset", "type", "tao"]
    base = base[cols].copy()
    base.columns = [*base.columns[:-1], "volume"]
    return ratio, window, date, base


def compute_score2_breakdown(sc: pd.DataFrame):
    jj = ["wallet", "asset", "type", "volume"]
    ratio, window, date, base = _load_imm_inputs()

    d1 = base[base["type"] == 1].groupby(jj[:-1]).sum().reset_index()
    d2 = base[base["type"] == 2].groupby(jj[:-1]).sum().reset_index()

    nn = subtensor.all_subnets()
    kk = sc[sc["index"].isna()]["coldkey"].unique()
    kk = [ck for ck in kk if ck in d2["wallet"].unique()]
    d3 = pd.DataFrame(columns=jj)
    for ck in kk:
        sn = d2[d2["wallet"] == ck]["asset"].unique()
        for stake_info in subtensor.get_stake_info_for_coldkey(ck):
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

    dfz = sc["score"].sum() * ratio[1] / ratio[0]
    scored["score2"] = float("nan")
    if dfz and scored["total"].sum():
        scored["score2"] = dfz * scored["total"] / scored["total"].sum()
    scored["score2"] /= scored["count"]

    # Full metagraph final score (same merge as ETF.core.functions.score2).
    metagraph_final = sc.merge(scored[["uid", "score2"]], on="uid", how="left")
    metagraph_final.loc[metagraph_final["score2"].notna(), "score"] = metagraph_final["score2"]
    metagraph_final = metagraph_final.drop(columns=["score2"])

    rank_by_uid = _rank_by_uid_from_metagraph(metagraph_final)

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
    st_ui.set_page_config(page_title=DASHBOARD_TITLE, layout="wide")
    _start_score2_prefetch(NETUID)

    st_ui.markdown(
        f"""<div style="display:flex;align-items:center;gap:0.375rem;margin:0 0 0.25rem 0;">
    <img src="{HODL_LOGO_URL}" width="64" height="64" style="object-fit:contain;flex-shrink:0;" alt="" />
    <h1 style="margin:0;padding:0;line-height:1.15;font-size:calc(1.75rem + 0.5vw);font-weight:600;">{DASHBOARD_TITLE}</h1>
</div>""",
        unsafe_allow_html=True,
    )
    st_ui.markdown(
        f"[HODL Exchange]({HODL_EXCHANGE_URL}) · "
        f"[HODL ETF Miner Dashboard]({HODL_ETF_DASHBOARD_URL})"
    )
    st_ui.caption(
        "IMM volume → capped type-2 → wallet totals → score2 splits; "
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
    c1.metric("IMM window (days)", data["window"])
    c2.metric("Incentive ratio", str(data["ratio"]))
    c3.metric("IMM date key", data["date"])

    rank_by_uid = data.get("rank_by_uid")
    if rank_by_uid is None:
        mf = data.get("metagraph_final")
        rank_by_uid = _rank_by_uid_from_metagraph(mf) if mf is not None else pd.Series(dtype="int64")

    alloc = (
        data["scored_rows"]
        .sort_values(["score2", "uid"], ascending=[False, True], na_position="last")
        .drop(columns=["index"], errors="ignore")
        .reset_index(drop=True)
    )
    alloc.insert(0, "rank", alloc["uid"].map(rank_by_uid))
    st_ui.subheader("score2 allocations")
    st_ui.dataframe(
        alloc,
        use_container_width=True,
        hide_index=True,
        column_config=_column_config_for_df(alloc, _TOOLTIP_ALLOC),
    )

    with st_ui.expander("IMM pipeline (intermediate tables)", expanded=False):
        st_ui.markdown("`type=2` raw volume")
        d2r = data["d2_type2_raw"]
        st_ui.dataframe(
            d2r,
            use_container_width=True,
            hide_index=True,
            column_config=_column_config_for_df(d2r, _TOOLTIP_IMM_WALLET_ROW),
        )
        st_ui.markdown("On-chain stake used for capping")
        d3 = data["d3_chain_stake"]
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
        st_ui.markdown("`type=2` after min(volume, stake)")
        d2c = data["d2_type2_capped"]
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
        st_ui.markdown("Per-wallet totals → score2 denominator (validator: type-1 + capped type-2)")
        wt = data["wallet_totals"]
        st_ui.dataframe(
            wt,
            use_container_width=True,
            hide_index=True,
            column_config=_column_config_for_df(wt, _TOOLTIP_WALLET_TOTALS),
        )
        raw2 = data["raw"][data["raw"]["type"] == 2]
        st_ui.markdown("Raw IMM rows (`type=2` only)")
        st_ui.dataframe(
            raw2,
            use_container_width=True,
            hide_index=True,
            column_config=_column_config_for_df(raw2, _TOOLTIP_IMM_WALLET_ROW),
        )

    csv_bytes = alloc.to_csv(index=False).encode("utf-8")
    st_ui.download_button("Download score2 allocations (CSV)", csv_bytes, file_name="score2_allocations.csv")


if __name__ == "__main__":
    render()
