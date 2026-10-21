"""SentiMind — Product review sentiment & aspect intelligence (Streamlit + scikit-learn NLP)."""
import html
import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import auth
import model as M
import theme
from ui import badge, hero, kpis, result_card, sidebar_brand

APP = dict(
    name="SentiMind", short="Review intelligence", icon="💬", tagline="Understand every customer review — instantly.",
    description="NLP that reads reviews like a human: sentiment, aspects, emotions and explanations, for a single review or 10,000 at once.",
    features=["4 models: Logistic Regression, Linear SVM, Naive Bayes, Neural Net", "Word-level explanations & aspect-based sentiment", "Batch CSV analysis with downloadable results", "Brand dashboards and a live review stream"],
    card1="💬 “Camera is amazing 😍” → Positive 96%", card2="🔋 Battery · Negative trend ↓",
    demo_name="Bharath Kumar", demo_email="demo@sentimind.app", demo_password="Demo@1234",
    roles=["Analyst", "Product manager", "Marketer", "Student"],
)
COLORS = {"positive": "#10b981", "neutral": "#f59e0b", "negative": "#ef4444"}
EMO_ICON = {"joy": "😄", "anger": "😠", "disappointment": "😞", "trust": "🤝", "surprise": "😮"}
SAMPLES = [
    "Absolutely love this phone! The camera is superb and the display is brilliant 😍 Battery easily lasts two days.",
    "Worst purchase ever. The battery is pathetic and customer service never replied. Waste of money 😡",
    "Delivery was quick and packaging was fine. Sound is average, nothing special for the price.",
    "Great, another laptop that heats up in ten minutes. 🙄 The display is good though.",
    "Paisa vasool! Build quality mast hai, but the strap is uncomfortable after a few hours.",
]

st.set_page_config(page_title="SentiMind — Review Sentiment AI", page_icon="💬", layout="wide")
theme.inject("Violet")


@st.cache_data(show_spinner="Loading 7,000 product reviews…")
def load():
    return M.generate()


@st.cache_resource(show_spinner="Training NLP model…")
def get_model(name):
    return M.train(load(), name)


@st.cache_data(show_spinner=False)
def score_corpus(name):
    pipe, _ = get_model(name)
    d = load().copy()
    proba = pipe.predict_proba(d.text)
    d["pred"] = np.array(pipe.classes_)[proba.argmax(1)]
    d["score"] = proba[:, list(pipe.classes_).index("positive")] - proba[:, list(pipe.classes_).index("negative")]
    return d


if "user" not in st.session_state:
    auth.auth_page(APP)
    st.stop()

user = st.session_state.user
sidebar_brand(APP, user)
PAGES = ["✍️ Analyze a review", "📂 Batch analysis", "🏷️ Brand dashboard", "📡 Live review stream", "🤖 Model lab", "👤 Account"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
model_name = st.sidebar.selectbox("🧠 Model", list(M.MODELS), key="model_name")
theme.theme_controls()
if st.sidebar.button("Log out", width="stretch"):
    auth.logout()
pipe, metrics = get_model(model_name)


def highlight(text):
    parts = []
    for tok, sc in M.token_scores(pipe, text):
        t = html.escape(tok)
        if abs(sc) < 0.05:
            parts.append(t)
        else:
            c = "16,185,129" if sc > 0 else "239,68,68"
            a = min(0.85, 0.2 + abs(sc) * 2)
            parts.append(f'<mark style="background:rgba({c},{a:.2f});color:inherit;border-radius:6px;padding:1px 4px">{t}</mark>')
    return "".join(parts)


def analyze(text):
    p = pipe.predict_proba([text])[0]
    classes = list(pipe.classes_)
    lab = classes[int(np.argmax(p))]
    return lab, dict(zip(classes, p)), M.stars_from(p, classes)


# ------------------------------------------------------------------ Analyze
if page == PAGES[0]:
    hero("Analyze a review", f"Sentiment, explanation, aspects and emotions — powered by {model_name}", "✍️")
    st.session_state.setdefault("review", SAMPLES[0])
    st.markdown("**Try a sample:**")
    cols = st.columns(len(SAMPLES))
    for i, (c, s) in enumerate(zip(cols, SAMPLES)):
        if c.button(["😍 Delighted", "😡 Angry", "😐 Mixed", "🙄 Sarcastic", "🇮🇳 Hinglish"][i], key=f"s{i}", width="stretch"):
            st.session_state.review = s
    text = st.text_area("Review text", key="review", height=120)
    if text.strip():
        lab, probs, stars = analyze(text)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            emoji = {"positive": "😊", "neutral": "😐", "negative": "😠"}[lab]
            result_card("Overall sentiment", f"{emoji} {lab.capitalize()}", f"Confidence {probs[lab] * 100:.1f}% · estimated rating {'★' * int(round(stars))}{'☆' * (5 - int(round(stars)))} ({stars}/5)", COLORS[lab], "#1e1b4b", "🧠")
            fig = go.Figure(go.Bar(x=[probs[l] * 100 for l in M.LABELS], y=[l.capitalize() for l in M.LABELS], orientation="h", marker_color=[COLORS[l] for l in M.LABELS], text=[f"{probs[l] * 100:.1f}%" for l in M.LABELS], textposition="auto"))
            fig.update_layout(height=220, xaxis_range=[0, 100], title="Class probabilities")
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("##### 🔍 Why? Word-level explanation")
            st.markdown(f'<div class="card" style="font-size:1.08rem;line-height:2">{highlight(text)}</div>', unsafe_allow_html=True)
            st.caption("Green words push towards positive, red words towards negative (TF-IDF weight × model coefficient).")
            ex = M.explain(pipe, text)
            if ex:
                e = pd.DataFrame(ex, columns=["term", "impact"]).sort_values("impact")
                fig = px.bar(e, x="impact", y="term", orientation="h", color=e.impact > 0, color_discrete_map={True: "#10b981", False: "#ef4444"})
                fig.update_layout(height=300, showlegend=False, title="Top contributing terms")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Word-level explanations are available for linear models (Logistic Regression, Complement NB).")
        a1, a2 = st.columns([1.4, 1])
        with a1:
            st.markdown("##### 🧩 Aspect-based sentiment")
            asp = M.aspects_of(pipe, text)
            if asp:
                for a in asp:
                    st.markdown(f'<div class="card" style="margin-bottom:8px;border-left:5px solid {COLORS[a["sentiment"]]}"><b>{a["aspect"].capitalize()}</b> {badge(a["sentiment"], {"positive": "good", "neutral": "warn", "negative": "bad"}[a["sentiment"]])} <small>{a["confidence"] * 100:.0f}%</small><br><small>“{html.escape(a["sentence"])}”</small></div>', unsafe_allow_html=True)
            else:
                st.caption("No product aspects (battery, camera, price, delivery…) detected.")
        with a2:
            st.markdown("##### 🎭 Emotions")
            emo = M.emotions(text)
            if sum(emo.values()):
                fig = go.Figure(go.Scatterpolar(r=list(emo.values()) + [list(emo.values())[0]], theta=[f"{EMO_ICON[k]} {k}" for k in emo] + [f"{EMO_ICON[list(emo)[0]]} {list(emo)[0]}"], fill="toself"))
                fig.update_layout(height=300, polar=dict(radialaxis=dict(visible=True, showticklabels=False)))
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("No strong emotion words found.")
        st.session_state.setdefault("history", [])
        if not st.session_state.history or st.session_state.history[-1]["text"] != text:
            st.session_state.history.append({"text": text, "sentiment": lab, "confidence": round(probs[lab], 3), "stars": stars})
        with st.expander(f"🕘 Session history ({len(st.session_state.history)})"):
            st.dataframe(pd.DataFrame(st.session_state.history[::-1]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Batch
elif page == PAGES[1]:
    hero("Batch analysis", "Upload a CSV of reviews — get sentiment, confidence, stars and aspects for every row", "📂")
    up = st.file_uploader("CSV file with a text column", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
    else:
        st.caption("No file? Using 300 sample reviews.")
        df = load().sample(300, random_state=4)[["date", "brand", "text"]].reset_index(drop=True)
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]
    col = st.selectbox("Text column", text_cols, index=text_cols.index("text") if "text" in text_cols else 0)
    if st.button("🚀 Run analysis", type="primary") or "batch" in st.session_state:
        t0 = time.time()
        texts = df[col].astype(str)
        proba = pipe.predict_proba(texts)
        classes = list(pipe.classes_)
        out = df.copy()
        out["sentiment"] = np.array(classes)[proba.argmax(1)]
        out["confidence"] = proba.max(1).round(3)
        out["stars"] = [M.stars_from(p, classes) for p in proba]
        out["aspects"] = [", ".join(f"{a['aspect']}:{a['sentiment'][:3]}" for a in M.aspects_of(pipe, t)) for t in texts]
        st.session_state.batch = True
        dist = out.sentiment.value_counts()
        kpis([
            ("📄", "Reviews analysed", f"{len(out):,}", f"in {time.time() - t0:.1f}s"),
            ("😊", "Positive", f"{dist.get('positive', 0) / len(out) * 100:.0f}%", f"{dist.get('positive', 0)} reviews"),
            ("😠", "Negative", f"{dist.get('negative', 0) / len(out) * 100:.0f}%", f"{dist.get('negative', 0)} reviews"),
            ("⭐", "Avg predicted rating", f"{out.stars.mean():.2f}", "out of 5"),
        ])
        c1, c2 = st.columns(2)
        fig = px.pie(out, names="sentiment", hole=0.55, color="sentiment", color_discrete_map=COLORS, title="Sentiment split")
        c1.plotly_chart(fig, width="stretch")
        asp = out.aspects.str.split(", ").explode().dropna()
        asp = asp[asp.str.contains(":")].str.split(":", expand=True)
        if len(asp):
            asp.columns = ["aspect", "s"]
            fig = px.histogram(asp, y="aspect", color="s", barmode="stack", color_discrete_map={"pos": "#10b981", "neu": "#f59e0b", "neg": "#ef4444"}, title="Aspect mentions by sentiment")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            c2.plotly_chart(fig, width="stretch")
        f = st.multiselect("Filter", M.LABELS, default=M.LABELS)
        st.dataframe(out[out.sentiment.isin(f)], hide_index=True, width="stretch", column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1)})
        st.download_button("⬇️ Download results (CSV)", out.to_csv(index=False), "sentimind_results.csv", "text/csv", type="primary")

# ------------------------------------------------------------------ Brand dashboard
elif page == PAGES[2]:
    hero("Brand dashboard", "Sentiment trends, aspect heatmaps and voice-of-customer highlights", "🏷️")
    d = score_corpus(model_name)
    c1, c2 = st.columns(2)
    cat = c1.selectbox("Category", list(M.BRANDS))
    brands = c2.multiselect("Brands", M.BRANDS[cat], default=M.BRANDS[cat])
    sub = d[(d.category == cat) & d.brand.isin(brands)]
    nps = lambda s: (s.stars >= 5).mean() * 100 - (s.stars <= 2).mean() * 100
    kpis([
        ("💬", "Reviews", f"{len(sub):,}", f"{sub.verified.mean() * 100:.0f}% verified buyers"),
        ("😊", "Positive share", f"{(sub.pred == 'positive').mean() * 100:.0f}%", "model-predicted"),
        ("📣", "Net Promoter proxy", f"{nps(sub):+.0f}", "5★ minus 1–2★"),
        ("⭐", "Avg rating", f"{sub.stars.mean():.2f}", "customer stars"),
    ])
    w = sub.assign(week=sub.date.dt.to_period("W").dt.start_time).groupby(["week", "brand"]).score.mean().reset_index()
    w["smooth"] = w.groupby("brand").score.transform(lambda s: s.rolling(4, min_periods=1).mean())
    fig = px.line(w, x="week", y="smooth", color="brand", title="Net sentiment over time (4-week rolling, −1 to +1)")
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(height=380, yaxis_title="net sentiment")
    st.plotly_chart(fig, width="stretch")
    rows = []
    for r in sub.itertuples():
        for a, l in M.parse_aspects(r.aspects):
            rows.append({"brand": r.brand, "aspect": a, "v": {"positive": 1, "neutral": 0, "negative": -1}[l]})
    am = pd.DataFrame(rows).pivot_table(index="brand", columns="aspect", values="v", aggfunc="mean")
    c1, c2 = st.columns([1.4, 1])
    fig = px.imshow(am, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=-1, zmax=1, title="Aspect sentiment heatmap")
    fig.update_layout(height=380)
    c1.plotly_chart(fig, width="stretch")
    sc = sub.groupby("brand").agg(pos=("pred", lambda s: (s == "positive").mean() * 100), n=("text", "size"), stars=("stars", "mean")).reset_index()
    fig = px.scatter(sc, x="stars", y="pos", size="n", color="brand", text="brand", title="Brand positioning: rating vs positive share", size_max=50)
    fig.update_traces(textposition="top center")
    fig.update_layout(height=380, showlegend=False, xaxis_title="avg stars", yaxis_title="% positive")
    c2.plotly_chart(fig, width="stretch")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("##### 💚 Most positive")
        for r in sub.nlargest(4, "score").itertuples():
            st.markdown(f'<div class="card" style="margin-bottom:8px;border-left:5px solid #10b981"><b>{r.brand}</b> · {"★" * r.stars}<br><small>{html.escape(r.text)}</small></div>', unsafe_allow_html=True)
    with t2:
        st.markdown("##### ❤️‍🩹 Most negative")
        for r in sub.nsmallest(4, "score").itertuples():
            st.markdown(f'<div class="card" style="margin-bottom:8px;border-left:5px solid #ef4444"><b>{r.brand}</b> · {"★" * r.stars}<br><small>{html.escape(r.text)}</small></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ Live stream
elif page == PAGES[3]:
    hero("Live review stream", "Simulated real-time feed — every new review is classified the moment it arrives", "📡")
    d = load()
    st.session_state.setdefault("stream", [])
    c1, c2, c3 = st.columns([1, 1, 2])
    n = c1.number_input("Reviews per tick", 1, 20, 6)
    alert = c2.slider("Negative-spike alert (%)", 20, 80, 40)
    run = c3.toggle("▶️ Stream running", value=False, help="Turn on to receive a new batch of reviews every few seconds")
    if run:
        new = d.sample(int(n), random_state=int(time.time()) % 10000)
        proba = pipe.predict_proba(new.text)
        for (_, r), p in zip(new.iterrows(), proba):
            lab = pipe.classes_[p.argmax()]
            st.session_state.stream.insert(0, {"time": pd.Timestamp.now().strftime("%H:%M:%S"), "brand": r.brand, "text": r.text, "sentiment": lab, "conf": float(p.max())})
        st.session_state.stream = st.session_state.stream[:120]
    s = pd.DataFrame(st.session_state.stream)
    if len(s):
        recent = s.head(30)
        neg = (recent.sentiment == "negative").mean() * 100
        if neg >= alert:
            st.error(f"🚨 Negative spike: {neg:.0f}% of the last 30 reviews are negative (threshold {alert}%)")
        kpis([("📥", "Received", f"{len(s)}", "this session"), ("😊", "Positive (last 30)", f"{(recent.sentiment == 'positive').mean() * 100:.0f}%", ""), ("😠", "Negative (last 30)", f"{neg:.0f}%", ""), ("🎯", "Avg confidence", f"{recent.conf.mean() * 100:.0f}%", "")])
        f1, f2 = st.columns([1.4, 1])
        with f1:
            for r in s.head(10).itertuples():
                st.markdown(f'<div class="card" style="margin-bottom:8px;border-left:5px solid {COLORS[r.sentiment]}"><small>{r.time} · <b>{r.brand}</b></small> {badge(r.sentiment, {"positive": "good", "neutral": "warn", "negative": "bad"}[r.sentiment])}<br>{html.escape(r.text)}</div>', unsafe_allow_html=True)
        with f2:
            fig = px.histogram(s, x="brand", color="sentiment", color_discrete_map=COLORS, title="Session sentiment by brand")
            fig.update_layout(height=420, xaxis_tickangle=-40)
            st.plotly_chart(fig, width="stretch")
    if run:
        time.sleep(2.5)
        st.rerun()

# ------------------------------------------------------------------ Model lab
elif page == PAGES[4]:
    hero("Model lab", "Compare classifiers on a stratified 20% hold-out (1,400 reviews)", "🤖")
    res = {n: get_model(n)[1] for n in M.MODELS}
    tab = pd.DataFrame({n: {k: r[k] for k in ["Accuracy", "Macro F1", "Weighted F1"]} for n, r in res.items()}).T
    best = tab["Macro F1"].idxmax()
    st.markdown(f"🏆 Best macro-F1: {badge(best, 'good')} — currently using {badge(model_name)}", unsafe_allow_html=True)
    fig = px.bar(tab.reset_index().melt(id_vars="index"), x="index", y="value", color="variable", barmode="group", text_auto=".3f", title="Model comparison")
    fig.update_layout(height=360, yaxis_range=[0.8, 1], xaxis_title="")
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(metrics["cm"], x=M.LABELS, y=M.LABELS, text_auto=True, color_continuous_scale="Purples", title=f"Confusion matrix — {model_name}", labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
    with c2:
        rep = pd.DataFrame(metrics["report"]).T.loc[M.LABELS, ["precision", "recall", "f1-score", "support"]]
        st.markdown("##### Per-class report")
        st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}), width="stretch")
        from sklearn.metrics import roc_curve, auc
        fig = go.Figure()
        for i, c in enumerate(metrics["classes"]):
            fpr, tpr, _ = roc_curve(metrics["y_te"] == c, metrics["proba"][:, i])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{c} (AUC {auc(fpr, tpr):.3f})", line=dict(color=COLORS[c], width=2.5)))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
        fig.update_layout(height=300, title="One-vs-rest ROC", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, width="stretch")
    terms = M.top_terms(pipe)
    if terms:
        st.markdown("##### 🔑 Most predictive n-grams per class")
        cols = st.columns(3)
        for col, lab in zip(cols, M.LABELS):
            t = pd.DataFrame(terms[lab], columns=["term", "weight"])
            fig = px.bar(t.iloc[::-1], x="weight", y="term", orientation="h", title=lab.capitalize(), color_discrete_sequence=[COLORS[lab]])
            fig.update_layout(height=420)
            col.plotly_chart(fig, width="stretch")
    st.info("Pipeline: word 1–2-gram TF-IDF ⊕ character 3–5-gram TF-IDF (robust to typos, Hinglish and emojis) → classifier. 7% label noise and sarcasm are injected to mimic real data.")

# ------------------------------------------------------------------ Account
else:
    hero(f"Hi, {user['name'].split()[0]} 👋", f"{user['email']} · {user['role']} · member since {user['created'][:10]}", "👤")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🔒 Change password")
        with st.form("pw"):
            old = st.text_input("Current password", type="password")
            new = st.text_input("New password", type="password")
            if st.form_submit_button("Update password", type="primary"):
                try:
                    auth.change_password(user["email"], old, new)
                    st.success("Password updated")
                except ValueError as e:
                    st.error(str(e))
    with c2:
        d = load()
        st.markdown("##### 📚 Corpus")
        st.markdown(f'<div class="card">{len(d):,} reviews · {d.brand.nunique()} brands · {d.category.nunique()} categories · {len(M.ASPECTS)} aspects. Registered users: <b>{auth.user_count()}</b>.</div>', unsafe_allow_html=True)
        st.download_button("⬇️ Download corpus (CSV)", d.to_csv(index=False), "sentimind_reviews.csv", "text/csv")
