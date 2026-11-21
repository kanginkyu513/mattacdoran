import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ──────────────────────────────────────────────
# 한글 폰트 CSS
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
# CSV 읽기 (utf-8-sig → cp949)
# ──────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ──────────────────────────────────────────────
# 열 이름 표준화 (KeyError 방지)
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

    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

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
# 다운로드 helper (utf-8-sig)
# ──────────────────────────────────────────────
def make_download_df(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("잎의 개수와 지하부길이는 관련이 있을까?")


# ── 사이드바 ──
st.sidebar.header("📂 CSV 업로드")

env_file = st.sidebar.file_uploader("환경 데이터 (CSV)", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 (CSV)", type="csv")

school_name = "우리 학교"
if env_file is not None:
    school_name = os.path.splitext(env_file.name)[0]
    school_name = school_name.split("_")[0] if "_" in school_name else school_name
elif growth_file is not None:
    school_name = os.path.splitext(growth_file.name)[0]
    school_name = school_name.split("_")[0] if "_" in school_name else school_name

st.sidebar.info(f"🏫 분석 대상: **{school_name}**")

growth_metrics = st.sidebar.multiselect(
    "📏 생육 측정 항목 (그래프에 볼 항목)",
    ["잎 수", "지상부 길이", "지하부 길이", "생중량"],
    default=["잎 수", "지하부 길이"],
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

if env_df.empty and growth_df.empty:
    st.info("CSV 파일 2개를 모두 업로드하면 대시보드가 표시됩니다.")
    st.stop()


# ══════════════════════════════════════════════
# 탭 1: 실험 목적 — 개체의 지하부 길이·ph
# ══════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(
    ["탭1: 실험 목적 — 개체의 지하부 길이·ph",
     "탭2: 날짜별 Ph값의 막대그래프",
     "탭3: 잎수와 지하부 길이의 산점도"],
)

with tab1:
    try:
        st.header("📖 실험 목적")
        st.write(
            f"**{school_name}**에서 재배 중인 극지식물을 대상으로, "
            "개체별 **지하부 길이(뿌리 길이)**와 환경 지표 중 하나인 **pH**를 함께 살펴봅니다. "
            "지하부 길이가 길수록 물을 더 잘 흡수할 수 있고, pH는 식물이 영양분을 흡수하기 좋은 환경인지 보여주는 지표입니다."
        )
        st.write("---")

        # 데이터가 있으면 요약 정보 표시
        if not growth_df.empty:
            if "지하부 길이" in growth_df.columns:
                root_vals = growth_df["지하부 길이"].dropna()
                if len(root_vals) >= 2:
                    col1, col2 = st.columns(2)
                    col1.metric(
                        "🌱 지하부 길이 평균",
                        f"{root_vals.mean():.1f}mm",
                    )
                    col2.metric(
                        "🌱 지하부 길이 범위",
                        f"{root_vals.min():.0f} ~ {root_vals.max():.0f}mm",
                    )
                    st.caption(
                        "※ 평균: 모든 개체의 값을 더해 개체 수로 나눈 값. "
                        "범위: 가장 짧은 개체부터 가장 긴 개체까지의 차이."
                    )
                else:
                    st.info("지하부 길이 데이터가 충분하지 않습니다.")
            else:
                st.info("생육 데이터에 '지하부 길이' 열이 없어 이 탭의 생육 지표를 표시할 수 없습니다.")
        else:
            st.info("생육 결과 CSV를 업로드하면 지하부 길이 정보가 표시됩니다.")

        st.write("---")

        if not env_df.empty:
            if "ph" in env_df.columns:
                ph_vals = env_df["ph"].dropna()
                if len(ph_vals) >= 2:
                    col3, col4 = st.columns(2)
                    col3.metric("🧪 평균 pH", f"{ph_vals.mean():.2f}")
                    col4.metric("🧪 pH 범위", f"{ph_vals.min():.2f} ~ {ph_vals.max():.2f}")
                    st.caption(
                        "※ pH: 0~14 범위의 산성·중성·알칼리성 지표. "
                        "대부분의 식물은 pH 6~7 정도를 선호합니다."
                    )
                else:
                    st.info("pH 데이터가 충분하지 않습니다.")
            else:
                st.info("환경 데이터에 'ph' 열이 없어 pH 지표를 표시할 수 없습니다.")
        else:
            st.info("환경 CSV를 업로드하면 pH 정보가 표시됩니다.")

        st.write("---")

        # 개체번호는 분석에서 제외하되, 요약표에는 원본 값 그대로 표시
        if not growth_df.empty and "개체번호" in growth_df.columns and "지하부 길이" in growth_df.columns:
            st.subheader("📋 개체별 지하부 길이 요약")
            ids = growth_df["개체번호"].copy()
            root_series = growth_df["지하부 길이"]
            combined = pd.DataFrame({"개체번호": ids, "지하부 길이(mm)": root_series})
            combined = combined.dropna(subset=["지하부 길이(mm)"])
            if not combined.empty:
                st.dataframe(combined, width="stretch", hide_index=True)
            else:
                st.info("표시할 지하부 길이 데이터가 없습니다.")
        else:
            st.info("개체별 지하부 길이 요약을 표시하려면 생육 CSV에 '지하부 길이' 열이 필요합니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ══════════════════════════════════════════════
# 탭 2: 날짜별 Ph값의 막대그래프
# ══════════════════════════════════════════════
with tab2:
    try:
        if env_df.empty:
            st.info("🌡️ 환경 데이터 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            st.header("📅 날짜별 pH 변화 (막대그래프)")

            if "time" in env_df.columns:
                env_df["date"] = env_df["time"].dt.date

                if "ph" in env_df.columns:
                    daily_ph = env_df.groupby("date")["ph"].mean().dropna()
                    if len(daily_ph) >= 1:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Bar(
                                x=daily_ph.index,
                                y=daily_ph.values,
                                name="평균 pH",
                                marker=dict(color="steelblue"),
                            )
                        )
                        fig.update_layout(
                            title="날짜별 평균 pH (막대그래프)",
                            xaxis_title="날짜",
                            yaxis_title="pH",
                            font=dict(
                                family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            ),
                            margin=dict(l=40, r=20, t=40, b=40),
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("pH 데이터가 충분하지 않아 막대그래프를 표시할 수 없습니다.")
                else:
                    st.info("환경 데이터에 'ph' 열이 없어 날짜별 pH 그래프를 표시할 수 없습니다.")
            else:
                st.info("환경 데이터에 'time' 열이 없어 날짜별 그래프를 표시할 수 없습니다.")

            st.write("---")

            # 환경 요약 표 (있는 열만)
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
                    st.info("요약할 환경 데이터가 충분하지 않습니다.")

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
# 탭 3: 잎수와 지하부 길이의 산점도
# ══════════════════════════════════════════════
with tab3:
    try:
        if growth_df.empty:
            st.info("📊 생육 결과 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            st.header("🔗 잎 수와 지하부 길이의 관계")

            if "잎 수" in growth_df.columns and "지하부 길이" in growth_df.columns:
                leaf_vals = growth_df["잎 수"].dropna()
                root_vals = growth_df["지하부 길이"].dropna()

                common_idx = leaf_vals.index.intersection(root_vals.index)
                x_vals = leaf_vals.loc[common_idx]
                y_vals = root_vals.loc[common_idx]

                if len(x_vals) >= 2:
                    corr = x_vals.corr(y_vals)

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="markers",
                            name="개체별 (잎 수, 지하부 길이)",
                            marker=dict(size=10, color="steelblue"),
                        )
                    )

                    # 추세선 직접 추가 (np.polyfit)
                    xv = x_vals.values
                    yv = y_vals.values
                    slope, intercept = np.polyfit(xv, yv, 1)
                    x_line = np.linspace(xv.min(), xv.max(), 100)
                    y_line = slope * x_line + intercept
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name="추세선",
                            line=dict(color="red", dash="dash"),
                        )
                    )

                    fig.update_layout(
                        title=f"잎 수 vs 지하부 길이 (상관계수 r = {corr:.3f})",
                        xaxis_title="잎 수 (장)",
                        yaxis_title="지하부 길이 (mm)",
                        font=dict(
                            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        ),
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig, width="stretch")

                    st.caption(
                        f"※ 상관계수(r): {corr:.3f} — 1에 가까울수록 강한 양의 관계, "
                        "-1에 가까울수록 강한 음의 관계, 0에 가까우면 관계가 약함. "
                        "잎 수가 많을수록 지하부 길이도 함께 길어지면 양의 관계입니다."
                    )
                else:
                    st.info("잎 수와 지하부 길이 데이터가 부족해 산점도를 표시할 수 없습니다.")
            else:
                missing = []
                if "잎 수" not in growth_df.columns:
                    missing.append("잎 수")
                if "지하부 길이" not in growth_df.columns:
                    missing.append("지하부 길이")
                st.info(f"산점도를 그리려면 '{', '.join(missing)}' 열이 모두 필요합니다.")

            st.write("---")

            # 생육 측정 항목 선택 기반 막대그래프 (재사용 함수)
            def draw_bar_metric(metric_name):
                if metric_name not in growth_df.columns:
                    return False
                vals = growth_df[metric_name].dropna()
                if len(vals) == 0:
                    return False
                ids = growth_df["개체번호"].copy()
                fig_bar = go.Figure()
                fig_bar.add_trace(
                    go.Bar(
                        x=ids.loc[vals.index],
                        y=vals.values,
                        name=metric_name,
                        marker=dict(color="lightblue"),
                    )
                )
                fig_bar.update_layout(
                    title=f"개체별 {metric_name}",
                    xaxis_title="개체",
                    yaxis_title=metric_name,
                    font=dict(
                        family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                    ),
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_bar, width="stretch")
                return True

            st.subheader("📈 개체별 생육 비교")
            plotted = False
            for m in growth_metrics:
                if m in growth_df.columns:
                    if draw_bar_metric(m):
                        plotted = True
            if not plotted:
                st.info("선택한 생육 측정 항목이 데이터에 없어 그래프를 표시할 수 없습니다.")

            st.write("---")

            # expander: 생육 데이터 원본 + 다운로드
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
