import json
import threading
import time
import traceback
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st_ui

from ETF.core.constants import IMM_DATA, IMM_RATIO, IMM_WINDOW, NETUID
from ETF.core.functions import score1, st as subtensor

# Server-wide cache: same Streamlit process shares entries for all browser sessions.
# Multiple app replicas (e.g. K8s) each have their own cache unless you add Redis/DB.
_CACHE_TTL_SECONDS = 3600
_PREFETCH_INTERVAL_SECONDS = 3600


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

    return {
        "ratio": ratio,
        "window": window,
        "date": date,
        "raw": base,
        "d1_type1": d1,
        "d2_type2_raw": d2,
        "d3_chain_stake": d3,
        "d2_type2_capped": d2_capped,
        "wallet_totals": totals,
        "scored_rows": scored,
    }


@st_ui.cache_data(
    ttl=_CACHE_TTL_SECONDS,
    show_spinner="Fetching chain + IMM (then served from cache until TTL expires)…",
)
def load_score2_bundle(netuid: int) -> dict:
    """Heavy work: score1 metagraph + stake + IMM APIs. Cached globally per `netuid` on this server."""
    sc = score1(netuid=netuid)
    return compute_score2_breakdown(sc)


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
                load_score2_bundle(netuid)
            except Exception:
                traceback.print_exc()
            time.sleep(_PREFETCH_INTERVAL_SECONDS)

    t = threading.Thread(target=worker, name="score2-prefetch", daemon=True)
    t.start()
    return t


def render():
    st_ui.set_page_config(page_title="Score2 Dashboard", layout="wide")
    _start_score2_prefetch(NETUID)
    st_ui.title("score2")
    st_ui.caption(
        "IMM volume → capped type-2 → wallet totals → score2 splits. "
        f"Uses subnet {NETUID} metagraph context internally (same as ETF/core/functions.py). "
        f"Snapshot is cached on this server for all users ({_CACHE_TTL_SECONDS}s TTL); "
        f"a background job refreshes about every {_PREFETCH_INTERVAL_SECONDS}s.",
    )

    try:
        data = load_score2_bundle(NETUID)
    except Exception as exc:
        st_ui.error(f"Failed: {exc}")
        st_ui.code(traceback.format_exc())
        return

    c1, c2, c3 = st_ui.columns(3)
    c1.metric("IMM window (days)", data["window"])
    c2.metric("Incentive ratio", str(data["ratio"]))
    c3.metric("IMM date key", data["date"])

    alloc = data["scored_rows"].sort_values("score2", ascending=False)
    st_ui.subheader("score2 allocations")
    st_ui.dataframe(alloc, use_container_width=True)

    with st_ui.expander("IMM pipeline (intermediate tables)", expanded=False):
        st_ui.markdown("`type=1` volume")
        st_ui.dataframe(data["d1_type1"], use_container_width=True)
        st_ui.markdown("`type=2` raw volume")
        st_ui.dataframe(data["d2_type2_raw"], use_container_width=True)
        st_ui.markdown("On-chain stake used for capping")
        st_ui.dataframe(data["d3_chain_stake"], use_container_width=True)
        st_ui.markdown("`type=2` after min(volume, stake)")
        st_ui.dataframe(data["d2_type2_capped"], use_container_width=True)
        st_ui.markdown("Per-wallet totals → score2 denominator")
        st_ui.dataframe(data["wallet_totals"], use_container_width=True)
        st_ui.markdown("Raw IMM rows (wallet, asset, type, volume)")
        st_ui.dataframe(data["raw"], use_container_width=True)

    csv_bytes = alloc.to_csv(index=False).encode("utf-8")
    st_ui.download_button("Download score2 allocations (CSV)", csv_bytes, file_name="score2_allocations.csv")


if __name__ == "__main__":
    render()
