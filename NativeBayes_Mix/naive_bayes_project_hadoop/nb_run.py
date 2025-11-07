# -*- coding: utf-8 -*-
import os, re, argparse
import numpy as np, pandas as pd, jieba
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.pipeline import Pipeline

def clean_text(s):
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9%：:，,。.!！?？￥$（）()《》<>@#\-\+\*\/]', ' ', s)
    return s.strip()

def cut_cn(s): 
    return ' '.join(jieba.cut(clean_text(s), cut_all=False))

def load_stopwords(path):
    if not path: return None
    if isinstance(path, str) and path.lower() == 'english': return 'english'
    words = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for w in f:
                w = w.strip()
                if w: words.append(w)
    except FileNotFoundError:
        return None
    return words

def build_vectorizer(max_features, ngram, min_df, stopwords_path):
    sw = load_stopwords(stopwords_path)
    if sw is not None and not isinstance(sw, (list, str)):
        sw = list(sw)
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram, min_df=min_df, stop_words=sw)

def plot_cm(cm, labels, title, out_path):
    plt.figure(); plt.imshow(cm, interpolation='nearest'); plt.title(title); plt.colorbar()
    ticks = np.arange(len(labels)); plt.xticks(ticks, labels, rotation=45); plt.yticks(ticks, labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser(description="Naive Bayes Text Classification (CN/EN)")
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default='cnb', choices=['mnb','cnb','bnb'])
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--max_features', type=int, default=20000)
    ap.add_argument('--min_df', type=int, default=2)
    ap.add_argument('--ngram', type=int, nargs=2, default=[1,2])
    ap.add_argument('--use_chi2', action='store_true')
    ap.add_argument('--chi2_k', type=int, default=15000)
    ap.add_argument('--stopwords', default='./resources/stopwords.txt')
    ap.add_argument('--fig_dir', default='./figs')
    ap.add_argument('--save_pred', type=str, default=None)
    ap.add_argument('--save_report', type=str, default=None)
    ap.add_argument('--save_mis', type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    df = pd.read_csv(args.data).dropna(subset=['label','text']).astype({'label':str,'text':str})
    print("分词中..."); tqdm.pandas(desc="jieba.cut"); df['tokens'] = df['text'].progress_apply(cut_cn)

    Xtr_txt, Xte_txt, ytr, yte = train_test_split(df['tokens'].values, df['label'].values, test_size=0.2, random_state=42, stratify=df['label'])
    vec = build_vectorizer(args.max_features, tuple(args.ngram), args.min_df, args.stopwords)
    Xtr = vec.fit_transform(Xtr_txt); Xte = vec.transform(Xte_txt)

    if args.use_chi2:
        scores, _ = chi2(Xtr, ytr); import numpy as np
        idx = np.argsort(scores)[::-1][:min(args.chi2_k, Xtr.shape[1])]
        Xtr = Xtr[:, idx]; Xte = Xte[:, idx]

    if args.model=='mnb': clf,name = MultinomialNB(alpha=args.alpha, fit_prior=True), f"MNB_a{args.alpha}"
    elif args.model=='bnb': clf,name = BernoulliNB(alpha=args.alpha), f"BNB_a{args.alpha}"
    else: clf,name = ComplementNB(alpha=args.alpha, norm=True), f"ComplementNB_a{args.alpha}"

    clf.fit(Xtr, ytr); ypred = clf.predict(Xte)
    rep_txt = classification_report(yte, ypred, digits=4)
    print(f"\n=== {name} ==="); print(rep_txt)
    cm = confusion_matrix(yte, ypred, labels=np.unique(yte)); print('Confusion Matrix:\n', cm)

    if args.save_report:
        os.makedirs(os.path.dirname(args.save_report), exist_ok=True)
        with open(args.save_report, 'w', encoding='utf-8') as f: f.write(rep_txt)

    proba = clf.predict_proba(Xte) if hasattr(clf, "predict_proba") else None
    import pandas as pd
    out_df = pd.DataFrame({'text': Xte_txt, 'true': yte, 'pred': ypred})
    if proba is not None:
        for i, cls in enumerate(clf.classes_):
            out_df[f'proba_{cls}'] = proba[:, i]
    if args.save_pred:
        os.makedirs(os.path.dirname(args.save_pred), exist_ok=True)
        out_df.to_csv(args.save_pred, index=False, encoding='utf-8-sig')
    if args.save_mis:
        mis = out_df[out_df['true'] != out_df['pred']]
        os.makedirs(os.path.dirname(args.save_mis), exist_ok=True)
        mis.to_csv(args.save_mis, index=False, encoding='utf-8-sig')

    plot_cm(cm, list(np.unique(yte)), f"{name} Confusion Matrix", os.path.join(args.fig_dir, f"{name}_cm.png"))

    alphas = [0.1, 0.5, 1.0]; skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_m, f1_c = [], []
    for a in alphas:
        pipe_m = Pipeline([('tfidf', build_vectorizer(args.max_features, tuple(args.ngram), args.min_df, args.stopwords)), ('clf', MultinomialNB(alpha=a))])
        pipe_c = Pipeline([('tfidf', build_vectorizer(args.max_features, tuple(args.ngram), args.min_df, args.stopwords)), ('clf', ComplementNB(alpha=a))])
        f1_m.append(cross_val_score(pipe_m, df['tokens'], df['label'], cv=skf, scoring='f1_macro').mean())
        f1_c.append(cross_val_score(pipe_c, df['tokens'], df['label'], cv=skf, scoring='f1_macro').mean())
        print(f"alpha={a}: MNB={f1_m[-1]:.4f}  CNB={f1_c[-1]:.4f}")
    plt.figure(); plt.plot(alphas, f1_m, marker='o', label='MultinomialNB'); plt.plot(alphas, f1_c, marker='o', label='ComplementNB')
    plt.xlabel('alpha'); plt.ylabel('CV F1-macro'); plt.title('Alpha Sweep (5-fold)'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, 'alpha_sweep.png'), dpi=150); plt.close()

if __name__ == "__main__":
    main()
