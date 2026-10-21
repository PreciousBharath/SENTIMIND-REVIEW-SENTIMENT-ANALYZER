"""SentiMind NLP engine: synthetic review corpus, TF-IDF models, explanations, aspects and emotions."""
import re

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

LABELS = ["negative", "neutral", "positive"]

BRANDS = {
    "Smartphones": ["Nova X5", "Pixelon 8", "Zentra Pro", "Galaxia M3"],
    "Laptops": ["AeroBook 14", "Titan G15", "Swift Air", "CoreMax 16"],
    "Headphones": ["BassPods", "SonicWave 2", "AuraBuds", "Echo Max"],
    "Smartwatches": ["FitRing S", "PulseWatch 3", "Chrono Fit", "Vita Band"],
    "Kitchen": ["ChefMate Mixer", "AirCrisp Fryer", "BrewBuddy", "HeatWave Oven"],
}
ASPECTS = {
    "battery": ["battery", "charge", "charging", "backup", "battery life"],
    "display": ["display", "screen", "brightness", "resolution"],
    "camera": ["camera", "photos", "pictures", "selfie", "video quality"],
    "performance": ["performance", "speed", "lag", "fast", "slow", "processor", "heating"],
    "sound": ["sound", "audio", "bass", "noise cancellation", "mic"],
    "build": ["build", "quality", "design", "material", "finish", "durable"],
    "price": ["price", "value", "money", "worth", "expensive", "cheap"],
    "delivery": ["delivery", "packaging", "shipping", "courier", "arrived"],
    "service": ["service", "support", "warranty", "refund", "replacement"],
    "comfort": ["comfort", "comfortable", "fit", "strap", "weight"],
}
NOUNS = {
    "battery": ["battery", "battery life", "charging speed", "battery backup"],
    "display": ["display", "screen", "brightness"],
    "camera": ["camera", "photo quality", "selfie camera", "video quality"],
    "performance": ["performance", "speed", "processor"],
    "sound": ["sound", "audio", "bass", "noise cancellation", "mic"],
    "build": ["build quality", "design", "material", "finish"],
    "price": ["price", "value for money"],
    "delivery": ["delivery", "packaging", "shipping"],
    "service": ["customer service", "support", "warranty service"],
    "comfort": ["comfort", "fit", "strap"],
}
SARCASM = [
    "Great, another {w} that gives up in two hours. 🙄",
    "Wow, the {w} is so good that I had to replace it within a week.",
    "Just what I wanted — a {w} that stops working on day three.",
    "Five stars for the box, zero for the {w}.",
]
CAT_ASPECTS = {
    "Smartphones": ["battery", "display", "camera", "performance", "price", "delivery", "service", "build"],
    "Laptops": ["battery", "display", "performance", "build", "price", "delivery", "service"],
    "Headphones": ["sound", "battery", "comfort", "build", "price", "delivery", "service"],
    "Smartwatches": ["battery", "display", "comfort", "build", "price", "service"],
    "Kitchen": ["performance", "build", "price", "delivery", "service"],
}
POS = ["amazing", "excellent", "superb", "great", "fantastic", "awesome", "outstanding", "brilliant", "impressive", "solid", "mast", "zabardast"]
NEG = ["terrible", "awful", "horrible", "disappointing", "poor", "pathetic", "useless", "worst", "bad", "bekar", "faltu"]
NEU = ["okay", "average", "decent", "fine", "acceptable", "normal", "standard"]
INTENS = ["", "", "really ", "very ", "absolutely ", "super ", "quite "]
POS_EMO = ["😍", "👍", "🔥", "❤️", "😊", ""]
NEG_EMO = ["😡", "👎", "😤", "😞", ""]
OPENERS = {
    "positive": ["Loved it!", "Totally worth it.", "Best purchase this year.", "Highly recommend.", "Exceeded my expectations.", "Paisa vasool!"],
    "negative": ["Very disappointed.", "Do not buy.", "Waste of money.", "Regret buying this.", "Returning it.", "Not as advertised."],
    "neutral": ["It is okay.", "Mixed feelings.", "Does the job.", "Nothing special.", "As expected for the price.", "Average product."],
}
EMOTIONS = {
    "joy": ["love", "loved", "happy", "amazing", "awesome", "fantastic", "great", "superb", "mast", "😍", "😊", "❤️", "delighted"],
    "anger": ["worst", "pathetic", "angry", "horrible", "useless", "faltu", "😡", "😤", "fraud", "cheated"],
    "disappointment": ["disappointed", "disappointing", "regret", "expected better", "not worth", "waste", "😞", "returning"],
    "trust": ["recommend", "reliable", "durable", "genuine", "worth", "trust", "solid"],
    "surprise": ["surprisingly", "unexpected", "wow", "shocked", "exceeded"],
}


def _sentence(rng, aspect, label):
    word = rng.choice(NOUNS[aspect])
    if label == "negative" and rng.random() < 0.12:
        return rng.choice(SARCASM).format(w=word)
    if label == "positive":
        adj = rng.choice(INTENS) + rng.choice(POS)
        tpl = rng.choice(["The {w} is {a}.", "{W} is {a} {e}", "Really happy with the {w}, {a}.", "{a_cap} {w}!", "The {w} is not bad at all, actually {a}."])
    elif label == "negative":
        adj = rng.choice(INTENS) + rng.choice(NEG)
        tpl = rng.choice(["The {w} is {a}.", "{W} is {a} {e}", "Not happy with the {w}, it is {a}.", "{a_cap} {w}.", "The {w} is not good, honestly {a}."])
    else:
        adj = rng.choice(NEU)
        tpl = rng.choice(["The {w} is {a}.", "{W} is {a}, nothing great.", "{W} is {a} for daily use."])
    e = rng.choice(POS_EMO if label == "positive" else NEG_EMO if label == "negative" else [""])
    return tpl.format(w=word, W=word.capitalize(), a=adj, a_cap=adj.strip().capitalize(), e=e).replace("  ", " ")


def generate(n=7000, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=364)
    brand_bias = {b: rng.normal(0, 0.35) for cat in BRANDS.values() for b in cat}
    for i in range(n):
        cat = rng.choice(list(BRANDS))
        brand = rng.choice(BRANDS[cat])
        day = int(rng.integers(0, 365))
        drift = 0.25 * np.sin(day / 58) if brand in ("Zentra Pro", "Titan G15") else 0
        p = np.array([0.30, 0.18, 0.52]) * np.exp(np.array([-1, 0, 1]) * (brand_bias[brand] + drift))
        label = rng.choice(LABELS, p=p / p.sum())
        k = int(rng.integers(1, 4))
        aspects = list(rng.choice(CAT_ASPECTS[cat], size=k, replace=False))
        sents, asp_labels = [], {}
        for j, a in enumerate(aspects):
            # mixed reviews: sometimes one aspect disagrees with overall label
            al = label if (j == 0 or rng.random() > 0.25) else rng.choice(LABELS)
            asp_labels[a] = al
            sents.append(_sentence(rng, a, al))
        text = " ".join([rng.choice(OPENERS[label])] + sents) if rng.random() > 0.2 else " ".join(sents)
        if rng.random() < 0.07:  # annotation noise, as in real-world labels
            label = rng.choice([l for l in LABELS if l != label])
        stars = {"positive": rng.choice([4, 5], p=[0.35, 0.65]), "neutral": 3, "negative": rng.choice([1, 2], p=[0.6, 0.4])}[label]
        rows.append({"date": start + pd.Timedelta(days=day), "category": cat, "brand": brand, "text": text, "label": label, "stars": int(stars),
                     "aspects": ",".join(f"{a}:{l}" for a, l in asp_labels.items()), "verified": bool(rng.random() > 0.15)})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _vec():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=40000)),
    ])


MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, C=4.0),
    "Linear SVM (calibrated)": lambda: CalibratedClassifierCV(LinearSVC(C=0.6), cv=3),
    "Complement Naive Bayes": lambda: ComplementNB(alpha=0.4),
    "Neural Net (MLP)": lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=25, early_stopping=True, random_state=0),
}


def train(df, name="Logistic Regression"):
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    pipe = Pipeline([("vec", _vec()), ("clf", MODELS[name]())])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)
    return pipe, {
        "Accuracy": accuracy_score(y_te, pred),
        "Macro F1": f1_score(y_te, pred, average="macro"),
        "Weighted F1": f1_score(y_te, pred, average="weighted"),
        "report": classification_report(y_te, pred, output_dict=True),
        "cm": confusion_matrix(y_te, pred, labels=LABELS),
        "y_te": y_te.values, "proba": proba, "classes": list(pipe.classes_),
    }


STOP = set("a an the is are was were it this that and or but to of for in on with my i me so at be as by its it's has have had".split()) | set("!.,?;:'\"-")


def _informative(term):
    toks = term.split()
    return not all(t in STOP for t in toks)


def explain(pipe, text, top=12):
    """Per-word contribution for linear models: tfidf weight × class coefficient (positive class − negative class)."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    word_vec = pipe.named_steps["vec"].transformer_list[0][1]
    vocab = word_vec.vocabulary_
    classes = list(clf.classes_)
    coef = clf.coef_[classes.index("positive")] - clf.coef_[classes.index("negative")]
    x = word_vec.transform([text])
    out = []
    for idx, val in zip(x.indices, x.data):
        term = next((t for t, i in vocab.items() if i == idx), None)
        if term is not None and _informative(term):
            out.append((term, float(val * coef[idx])))
    return sorted(out, key=lambda t: -abs(t[1]))[:top]


def token_scores(pipe, text):
    contrib = dict(explain(pipe, text, top=200))
    toks = re.findall(r"\w+|[^\w\s]|\s+", text)
    return [(t, contrib.get(t.lower(), 0.0)) for t in toks]


def aspects_of(pipe, text):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    found = []
    for s in sents:
        low = s.lower()
        for a, kws in ASPECTS.items():
            if any(re.search(rf"\b{re.escape(k)}\b", low) for k in kws):
                p = pipe.predict_proba([s])[0]
                lab = pipe.classes_[int(np.argmax(p))]
                found.append({"aspect": a, "sentence": s, "sentiment": lab, "confidence": float(p.max())})
    return found


def emotions(text):
    low = text.lower()
    return {e: sum(low.count(w) for w in ws) for e, ws in EMOTIONS.items()}


def stars_from(proba, classes):
    p = dict(zip(classes, proba))
    return round(1 + 4 * (p.get("positive", 0) + 0.5 * p.get("neutral", 0)), 1)


def top_terms(pipe, n=15):
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return {}
    vec = pipe.named_steps["vec"].transformer_list[0][1]
    names = np.array(vec.get_feature_names_out())
    k = len(names)
    out = {}
    for i, c in enumerate(clf.classes_):
        w = clf.coef_[i][:k]
        idx = np.argsort(w)[-n:][::-1]
        out[c] = list(zip(names[idx], w[idx]))
    return out


def parse_aspects(s):
    return [tuple(x.split(":")) for x in s.split(",") if ":" in x]
