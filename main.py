import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ──────────────────────────────────────────────
# 1. 한글 폰트 CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 2. CSV 읽기 (인코딩 대응)
# ──────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    """utf-8-sig → cp949 순서로 시도, 실패하면 빈 DataFrame"""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ──────────────────────────────────────────────
# 3. 열 이름 표준화 (KeyError 방지)
# ──────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)

    # 숫자 변환 (글자 열은 계산에서 제외)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    # 생중량이 없고 지상부·지하부 질량이 있으면 합산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    # time 파싱
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed

    return df


# ──────────────────────────────────────────────
# 4. 추세선 helper (np.polyfit)
# ──────────────────────────────────────────────
def add_trendline(fig, x, y, color="red", dash="dash", name="추세선"):
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return
    xv, yv = valid.iloc[:, 0], valid.iloc[:, 1]
    slope, intercept = np.polyfit(xv, yv, 1)
    x_line = np.linspace(xv.min(), xv.max(), 100)
    y_line = slope * x_line + intercept
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=name,
            line=dict(color=color, dash=dash),
        )
    )


# ──────────────────────────────────────────────
# 5. 다운로드 helper (utf-8-sig)
# ──────────────────────────────────────────────
def make_download_df(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ──────────────────────────────────────────────
# 메인 앱
# ──────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")


# ── 사이드바: 파일 업로드 ──
st.sidebar.header("📂 CSV 업로드")

env_file = st.sidebar.file_uploader("환경 데이터 (CSV)", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 (CSV)", type="csv")

# 학교명 추출 (파일명 앞부분)
school_name = "우리 학교"
if env_file is not None:
    school_name = os.path.splitext(env_file.name)[0]
    school_name = school_name.split("_")[0] if "_" in school_name else school_name
elif growth_file is not None:
    school_name = os.path.splitext(growth_file.name)[0]
    school_name = school_name.split("_")[0] if "_" in school_name else school_name

st.sidebar.info(f"🏫 분석 대상: **{school_name}**")

# 생육 측정 항목 선택
growth_metrics = st.sidebar.multiselect(
    "📏 생육 측정 항목 (그래프로 볼 항목)",
    ["잎 수", "지상부 길이", "지하부 길이", "생중량"],
    default=["생중량", "잎 수"],
)

# ── 데이터 로딩 (한 번만) ──
env_df = pd.DataFrame()
growth_df = pd.DataFrame()

if env_file is not None:
    with st.spinner("환경 데이터를 읽는 중..."):
        env_df = read_csv(env_file)
        env_df = std_cols(env_df)

if growth_file is not None:
    with st.spinner("생육 결과 데이터를 읽는 중..."):
        growth_df = read_csv(growth_file)
        growth_df = std_cols(growth_df)

# 데이터 없음 안내
if env_df.empty and growth_df.empty:
    st.info("CSV 파일 2개를 모두 업로드하면 대시보드가 표시됩니다.")
    st.stop()


# ══════════════════════════════════════════════
# 탭 1: 실험 개요
# ══════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.header("연구 배경 및 목적")
        st.write(
            f"**{school_name}**에서 재배 중인 극지식물의 **개체별 생육 차이**를 분석합니다. "
            "온도·습도·pH·EC 등 환경 데이터와 각 개체의 잎 수, 줄기·뿌리 길이, 생체중량(생중량)을 비교하여 "
            "어떤 개체가 잘 자랐는지, 환경과 생육 사이에 어떤 관계가 있는지 살펴봅니다."
        )
        st.write("---")

        # ── 데이터 요약 표 ──
        summary_rows = []
        if not env_df.empty:
            times = env_df["time"].dropna()
            if len(times) > 0:
                start = times.min().strftime("%Y-%m-%d")
                end = times.max().strftime("%Y-%m-%d")
                summary_rows.append(["측정 기간", f"{start} ~ {end}"])
            else:
                summary_rows.append(["측정 기간", "알 수 없음"])
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            summary_rows.append(["환경 측정 항목", f"{len(env_cols)}개 ({', '.join(env_cols)})"])
        else:
            summary_rows.append(["측정 기간", "환경 CSV를 업로드하세요"])
            summary_rows.append(["환경 측정 항목", "환경 CSV를 업로드하세요"])

        if not growth_df.empty:
            n_individuals = growth_df["개체번호"].nunique()
            summary_rows.append(["총 개체 수", f"{n_individuals}개체"])
            growth_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
            summary_rows.append(["생육 측정 항목", f"{len(growth_cols)}개 ({', '.join(growth_cols)})"])
        else:
            summary_rows.append(["총 개체 수", "생육 CSV를 업로드하세요"])
            summary_rows.append(["생육 측정 항목", "생육 CSV를 업로드하세요"])

        st.subheader("📋 데이터 요약")
        st.dataframe(
            pd.DataFrame(summary_rows, columns=["항목", "내용"]),
            width="stretch",
            hide_index=True,
        )
        st.write("---")

        # ── 주요 지표 카드 4개 ──
        col1, col2, col3, col4 = st.columns(4)

        # 총 개체수
        if not growth_df.empty and "개체번호" in growth_df.columns:
            n = growth_df["개체번호"].nunique()
            col1.metric("🌱 총 개체수", f"{n}개체")
        else:
            col1.metric("🌱 총 개체수", "-")

        # 평균 온도
        if not env_df.empty and "temperature" in env_df.columns:
            avg_t = env_df["temperature"].dropna().mean()
            col2.metric("🌡️ 평균 온도", f"{avg_t:.1f}°C")
        else:
            col2.metric("🌡️ 평균 온도", "-")

        # 평균 습도
        if not env_df.empty and "humidity" in env_df.columns:
            avg_h = env_df["humidity"].dropna().mean()
            col3.metric("💧 평균 습도", f"{avg_h:.1f}%")
        else:
            col3.metric("💧 평균 습도", "-")

        # 평균 생중량
        if not growth_df.empty and "생중량" in growth_df.columns:
            avg_b = growth_df["생중량"].dropna().mean()
            col4.metric("⚖️ 평균 생중량", f"{avg_b:.2f}g")
        else:
            col4.metric("⚖️ 평균 생중량", "-")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ══════════════════════════════════════════════
# 탭 2: 환경 데이터
# ══════════════════════════════════════════════
with tab2:
    try:
        if env_df.empty:
            st.info("🌡️ 환경 데이터 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            st.header("🌡️ 환경 데이터 분석")

            # 환경 요약 표
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_cols:
                st.subheader("📊 환경 요약 (평균 · 최소 · 최대)")
                summary_data = {}
                for col in env_cols:
                    vals = env_df[col].dropna()
                    if len(vals) > 0:
                        summary_data[col] = {
                            "평균": round(vals.mean(), 2),
                            "최소": round(vals.min(), 2),
                            "최대": round(vals.max(), 2),
                        }
                if summary_data:
                    st.dataframe(
                        pd.DataFrame(summary_data).T,
                        width="stretch",
                        hide_index=False,
                    )
                else:
                    st.info("환경 데이터가 충분하지 않아 요약 표를 표시할 수 없습니다.")
            else:
                st.info("환경 데이터에 temperature, humidity, ph, ec 열이 없어 요약을 표시할 수 없습니다.")

            st.write("---")

            # 일별 시계열
            if "time" in env_df.columns:
                env_df["date"] = env_df["time"].dt.date

                st.subheader("📅 일별 환경 변화 (일별 평균)")

                plot_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
                if plot_cols:
                    n_plots = len(plot_cols)
                    fig = make_subplots(
                        rows=n_plots, cols=1,
                        shared_xaxes=True,
                        subplot_titles=[c for c in plot_cols],
                    )

                    for i, col in enumerate(plot_cols, start=1):
                        daily = env_df.groupby("date")[col].mean().dropna()
                        if len(daily) == 0:
                            continue
                        fig.add_trace(
                            go.Scatter(
                                x=daily.index,
                                y=daily.values,
                                mode="lines+markers",
                                name=col,
                                line=dict(width=2),
                                marker=dict(size=5),
                            ),
                            row=i, col=1,
                        )

                    fig.update_layout(
                        height=200 * n_plots + 50,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=False,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    fig.update_xaxes(title_text="날짜")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("그래프로 표시할 환경 열이 없습니다.")

            # expander: 원본 테이블 + 다운로드
            with st.expander("📄 환경 데이터 원본 보기 및 다운로드"):
                st.dataframe(env_df, width="stretch", hide_index=True)
                csv_bytes = make_download_df(env_df)
                st.download_button(
                    label="CSV 다운로드 (utf-8-sig)",
                    data=csv_bytes,
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ══════════════════════════════════════════════
# 탭 3: 생육 결과
# ══════════════════════════════════════════════
with tab3:
    try:
        if growth_df.empty:
            st.info("📊 생육 결과 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            st.header("📊 생육 결과 분석")

            # ── 🥇 핵심 결과 카드 ──
            if "생중량" in growth_df.columns:
                bw = growth_df["생중량"].dropna()
                if len(bw) > 0:
                    mean_bw = bw.mean()
                    max_idx = bw.idxmax()
                    min_idx = bw.idxmin()
                    max_val = bw.loc[max_idx]
                    min_val = bw.loc[min_idx]

                    # 개체번호는 원래 값 그대로 표시 (float 변환 방지)
                    id_col = growth_df["개체번호"]
                    max_id = id_col.loc[max_idx] if max_idx in id_col.index else "?"
                    min_id = id_col.loc[min_idx] if min_idx in id_col.index else "?"

                    card1, card2, card3 = st.columns(3)
                    card1.metric("🏆 평균 생중량", f"{mean_bw:.2f}g")
                    card2.metric("🔝 최대 생중량", f"{max_val:.2f}g (개체 {max_id})")
                    card3.metric("🔻 최소 생중량", f"{min_val:.2f}g (개체 {min_id})")
                else:
                    st.metric("🏆 평균 생중량", "-")
            else:
                st.info("생중량 열이 없어 핵심 결과 카드를 표시할 수 없습니다.")

            st.write("---")

            # ── 개체별 생육 비교 (막대 그래프) ──
            st.subheader("📈 개체별 생육 비교")

            plot_metrics = [m for m in growth_metrics if m in growth_df.columns]
            if plot_metrics:
                # 개체번호 보존 (float 방지)
                id_vals = growth_df["개체번호"].copy()

                for metric in plot_metrics:
                    fig = go.Figure()
                    vals = growth_df[metric].dropna()
                    if len(vals) == 0:
                        continue

                    fig.add_trace(
                        go.Bar(
                            x=id_vals.loc[vals.index],
                            y=vals.values,
                            name=metric,
                            marker=dict(color="lightblue"),
                        )
                    )

                    fig.update_layout(
                        title=f"개체별 {metric}",
                        xaxis_title="개체",
                        yaxis_title=metric,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("선택한 생육 측정 항목이 데이터에 없어 그래프를 표시할 수 없습니다.")

            st.write("---")

            # ── 생중량 분포 (박스플롯) ──
            if "생중량" in growth_df.columns:
                bw = growth_df["생중량"].dropna()
                if len(bw) >= 2:
                    fig_box = go.Figure()
                    fig_box.add_trace(
                        go.Box(
                            y=bw.values,
                            name="생중량",
                            fillcolor="lightblue",
                            marker=dict(color="lightblue"),
                        )
                    )
                    fig_box.update_layout(
                        title="생중량 분포 (박스플롯)",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_box, width="stretch")

                    # 평균·표준편차·CV
                    mean_v = bw.mean()
                    std_v = bw.std()
                    cv_v = (std_v / mean_v * 100) if mean_v != 0 else 0
                    st.write(
                        f"**평균**: {mean_v:.2f}g &nbsp;|&nbsp; "
                        f"**표준편차**: {std_v:.2f}g &nbsp;|&nbsp; "
                        f"**변동계수(CV)**: {cv_v:.1f}%"
                    )
                    st.caption(
                        "※ 표준편차: 값이 평균에서 얼마나 퍼져 있는지 나타내는 수치. "
                        "값이 클수록 개체 간 차이가 큼. "
                        "※ 변동계수(CV): 표준편차를 평균으로 나눈 백분율 — 서로 다른 단위·규모끼리 비교할 때 유용."
                    )
                else:
                    st.info("생중량 데이터가 충분하지 않아 박스플롯을 표시할 수 없습니다.")
            else:
                st.info("생중량 열이 없어 분포 그래프를 표시할 수 없습니다.")

            st.write("---")

            # ── 상관관계 분석 (산점도 + 추세선) ──
            st.subheader("🔗 상관관계 분석")

            scatter_pairs = []
            if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                scatter_pairs.append(("잎 수", "잎 수", "생중량"))
            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                scatter_pairs.append(("지상부 길이", "지상부 길이", "생중량"))

            if scatter_pairs:
                for label, x_col, y_col in scatter_pairs:
                    x_data = growth_df[x_col].dropna()
                    y_data = growth_df[y_col].dropna()
                    common_idx = x_data.index.intersection(y_data.index)
                    x_vals = x_data.loc[common_idx]
                    y_vals = y_data.loc[common_idx]

                    if len(x_vals) >= 2:
                        corr = x_vals.corr(y_vals)
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=y_vals,
                                mode="markers",
                                name=f"{label} vs 생중량",
                                marker=dict(size=10, color="steelblue"),
                            )
                        )
                        add_trendline(fig_scatter, x_vals, y_vals, color="red", dash="dash")

                        fig_scatter.update_layout(
                            title=f"{label} vs 생중량 (상관계수 r = {corr:.3f})",
                            xaxis_title=label,
                            yaxis_title="생중량 (g)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=40, r=20, t=40, b=40),
                        )
                        st.plotly_chart(fig_scatter, width="stretch")
                        st.caption(
                            f"※ 상관계수(r): {corr:.3f} — 1에 가까울수록 강한 양의 관계, "
                            "-1에 가까울수록 강한 음의 관계, 0에 가까우면 관계가 약함."
                        )
                    else:
                        st.info(f"{label} vs 생중량: 데이터가 부족해 상관분석을 건너뜁니다.")
            else:
                st.info("잎 수·지상부 길이와 생중량을 함께 비교할 데이터가 없습니다.")

            st.write("---")

            # ── 생중량 상위 25% vs 하위 25% 비교 ──
            if "생중량" in growth_df.columns:
                bw = growth_df["생중량"].dropna()
                if len(bw) >= 4:
                    q75 = bw.quantile(0.75)
                    q25 = bw.quantile(0.25)
                    top_ids = growth_df.loc[bw[bw >= q75].index, "개체번호"]
                    bottom_ids = growth_df.loc[bw[bw <= q25].index, "개체번호"]

                    st.subheader("🏅 생중량 상위 25% vs 하위 25% 비교")
                    top_stats = growth_df.loc[bw[bw >= q75].index].dropna()
                    bottom_stats = growth_df.loc[bw[bw <= q25].index].dropna()

                    comp_rows = []
                    for m in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]:
                        if m in growth_df.columns:
                            t_vals = top_stats[m].dropna()
                            b_vals = bottom_stats[m].dropna()
                            if len(t_vals) > 0 and len(b_vals) > 0:
                                comp_rows.append({
                                    "측정 항목": m,
                                    "상위 25% 평균": round(t_vals.mean(), 2),
                                    "하위 25% 평균": round(b_vals.mean(), 2),
                                    "차이": round(t_vals.mean() - b_vals.mean(), 2),
                                })

                    if comp_rows:
                        st.dataframe(
                            pd.DataFrame(comp_rows),
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.info("비교 가능한 측정 항목이 없습니다.")
                else:
                    st.info("개체 수가 적어 상위·하위 25% 비교를 생략합니다.")

            # expander: 원본 생육 데이터 + 다운로드
            with st.expander("📄 생육 데이터 원본 보기 및 다운로드"):
                st.dataframe(growth_df, width="stretch", hide_index=True)
                csv_bytes = make_download_df(growth_df)
                st.download_button(
                    label="CSV 다운로드 (utf-8-sig)",
                    data=csv_bytes,
                    file_name=f"{school_name}_생육결과.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
