import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Portfolio Optimizer",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f0f2f5; }
    [data-testid="stSidebar"] { background-color: #0a1628; }
    [data-testid="stSidebar"] * { color: #c8d6e5 !important; }
    .top-bar {
        background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 100%);
        padding: 20px 32px;
        margin: -24px -24px 24px -24px;
        border-bottom: 2px solid #c9a84c;
    }
    .top-bar-title { font-size: 20px; font-weight: 600; color: white; }
    .top-bar-sub { font-size: 12px; color: #7f9abd; margin-top: 4px; }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #0a1628;
        border-radius: 2px;
        padding: 16px 20px;
    }
    .metric-label { font-size: 10px; font-weight: 600; color: #7f8c9a; letter-spacing: 1.5px; text-transform: uppercase; }
    .metric-value { font-size: 22px; font-weight: 600; color: #1a2b4a; margin-top: 4px; }
    .section-label { font-size: 10px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #4a6fa5 !important; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

now = datetime.now().strftime("%Y.%m.%d")
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-title">Portfolio Optimization System</div>
    <div class="top-bar-sub">Modern Portfolio Theory &nbsp;|&nbsp; Efficient Frontier &nbsp;|&nbsp; Monte Carlo Simulation &nbsp;|&nbsp; {now}</div>
</div>
""", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.markdown('<div style="padding:20px 0 16px;border-bottom:1px solid #1e3a5f;margin-bottom:20px;"><div style="font-size:15px;font-weight:600;color:white;">Portfolio Optimizer</div><div style="font-size:10px;color:#4a6fa5;letter-spacing:2px;text-transform:uppercase;margin-top:2px;">Quantitative Finance</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Ticker Symbols</div>', unsafe_allow_html=True)
    default_tickers = "AAPL, MSFT, GOOGL, JPM, BRK-B, JNJ, XOM"
    ticker_input = st.text_area("", value=default_tickers, height=100, label_visibility="collapsed")

    st.markdown('<div class="section-label" style="margin-top:16px;">Analysis Period</div>', unsafe_allow_html=True)
    start_date = st.date_input("", value=datetime(2020, 1, 1), label_visibility="collapsed")
    end_date = st.date_input("", value=datetime(2024, 12, 31), label_visibility="collapsed")

    st.markdown('<div class="section-label" style="margin-top:16px;">Parameters</div>', unsafe_allow_html=True)
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.1) / 100
    num_portfolios = st.select_slider("Monte Carlo Simulations", options=[1000, 5000, 10000, 20000], value=10000)

    st.markdown('<br>', unsafe_allow_html=True)
    run_btn = st.button("Run Optimization", use_container_width=True, type="primary")

# 메인
if not run_btn:
    st.markdown("""
    <div style="text-align:center;padding:80px 0;color:#a0aec0;">
        <div style="font-size:15px;font-weight:500;color:#4a5568;margin-bottom:8px;">Enter ticker symbols and click Run Optimization</div>
        <div style="font-size:12px;">Supports any Yahoo Finance ticker — stocks, ETFs, indices</div>
    </div>
    """, unsafe_allow_html=True)
else:
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

    with st.spinner("Fetching market data..."):
        try:
            df = yf.download(tickers, start=start_date, end=end_date)['Close']
            if len(tickers) == 1:
                df = df.to_frame(name=tickers[0])
            df = df.dropna(axis=1, how='all')
            df = df.fillna(method='ffill').dropna()
            valid_tickers = list(df.columns)
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()

    returns = df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(valid_tickers)

    def portfolio_performance(weights):
        ret = np.sum(mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol
        return ret, vol, sharpe

    # Monte Carlo
    with st.spinner(f"Running {num_portfolios:,} Monte Carlo simulations..."):
        results = np.zeros((3, num_portfolios))
        weights_record = []
        np.random.seed(42)
        for i in range(num_portfolios):
            w = np.random.random(num_assets)
            w /= np.sum(w)
            weights_record.append(w)
            r, v, s = portfolio_performance(w)
            results[0, i] = v
            results[1, i] = r
            results[2, i] = s

    # 최적화
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_w = np.array([1/num_assets] * num_assets)

    opt_sharpe = minimize(lambda w: -portfolio_performance(w)[2], init_w,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    opt_vol = minimize(lambda w: portfolio_performance(w)[1], init_w,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    ms_ret, ms_vol, ms_sharpe = portfolio_performance(opt_sharpe.x)
    mv_ret, mv_vol, mv_sharpe = portfolio_performance(opt_vol.x)

    # 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Max Sharpe Ratio</div><div class="metric-value">{ms_sharpe:.3f}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Expected Return</div><div class="metric-value">{ms_ret*100:.1f}%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Min Volatility</div><div class="metric-value">{mv_vol*100:.1f}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Assets Analyzed</div><div class="metric-value">{num_assets}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 효율적 프론티어
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("**Efficient Frontier**")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        scatter = ax.scatter(results[0]*100, results[1]*100,
                            c=results[2], cmap='viridis', alpha=0.4, s=8)
        plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
        ax.scatter(ms_vol*100, ms_ret*100, marker='*', color='#c9a84c', s=600, zorder=5,
                  label=f'Max Sharpe ({ms_sharpe:.2f})', edgecolors='black', linewidth=0.5)
        ax.scatter(mv_vol*100, mv_ret*100, marker='*', color='#e74c3c', s=600, zorder=5,
                  label=f'Min Volatility ({mv_vol*100:.1f}%)', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Annual Volatility (%)', fontsize=11)
        ax.set_ylabel('Annual Return (%)', fontsize=11)
        ax.set_title(f'Efficient Frontier — {num_portfolios:,} Simulations', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#fafafa')
        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        st.markdown("**Max Sharpe Portfolio Weights**")
        weights_df = pd.DataFrame({
            'Asset': valid_tickers,
            'Weight (%)': (opt_sharpe.x * 100).round(1)
        }).sort_values('Weight (%)', ascending=False)
        weights_df = weights_df[weights_df['Weight (%)'] > 0.1]

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(weights_df)))
        ax2.pie(weights_df['Weight (%)'], labels=weights_df['Asset'],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Max Sharpe Ratio Portfolio', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig2)

    # 상세 테이블
    st.markdown("<br>**Optimization Results**", unsafe_allow_html=True)
    result_df = pd.DataFrame({
        'Asset': valid_tickers,
        'Max Sharpe Weight (%)': (opt_sharpe.x * 100).round(2),
        'Min Vol Weight (%)': (opt_vol.x * 100).round(2),
        'Annual Return (%)': (mean_returns * 100).round(2),
        'Annual Volatility (%)': (returns.std() * np.sqrt(252) * 100).round(2),
        'Sharpe Ratio': ((mean_returns - risk_free_rate) / (returns.std() * np.sqrt(252))).round(3)
    })
    st.dataframe(result_df, use_container_width=True)

    st.markdown("""
    <div style="font-size:10px;color:#a0aec0;padding:12px 0;border-top:1px solid #e2e8f0;text-align:center;">
        This tool is for educational purposes only. Not financial advice.
    </div>
    """, unsafe_allow_html=True)