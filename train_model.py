import json
import os
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


LABELS = [
    'manipulative',
    'religious_manipulation',
    'political_manipulation',
    'anti_constitutional',
]


def build_vectorizer():
    # Combine word and character n-grams for better robustness on varied phrasing.
    return FeatureUnion([
        (
            'word_tfidf',
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=12000,
                sublinear_tf=True,
            ),
        ),
        (
            'char_tfidf',
            TfidfVectorizer(
                lowercase=True,
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=2,
                max_features=15000,
                sublinear_tf=True,
            ),
        ),
    ])


def model_probability(model, X_vec):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_vec)[:, 1]
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_vec)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X_vec).astype(float)


def get_group_cv_splits(y: pd.Series, groups: pd.Series, max_splits: int = 5):
    positive_count = int(y.sum())
    negative_count = int(len(y) - positive_count)
    minority_count = min(positive_count, negative_count)
    unique_groups = pd.Series(groups).nunique()
    return min(max_splits, minority_count, int(unique_groups))


def choose_model(X_train_vec, y: pd.Series, groups: pd.Series):
    # Compare simple linear models via grouped CV; pick the one with best mean F1.
    model_candidates = [
        ('logreg_c0.5_balanced', LogisticRegression(max_iter=3000, C=0.5, solver='liblinear', class_weight='balanced', random_state=42)),
        ('logreg_c1_balanced', LogisticRegression(max_iter=3000, C=1.0, solver='liblinear', class_weight='balanced', random_state=42)),
        ('logreg_c2_none', LogisticRegression(max_iter=3000, C=2.0, solver='liblinear', class_weight=None, random_state=42)),
        ('linearsvc_c0.5_balanced', LinearSVC(C=0.5, class_weight='balanced', random_state=42)),
        ('linearsvc_c1_balanced', LinearSVC(C=1.0, class_weight='balanced', random_state=42)),
    ]

    n_splits = get_group_cv_splits(y, groups)
    if n_splits < 2:
        fallback = LogisticRegression(max_iter=3000, C=1.0, solver='liblinear', class_weight='balanced', random_state=42)
        return 'logreg_group_fallback', fallback, None

    cv = GroupKFold(n_splits=n_splits)

    best_name = None
    best_model = None
    best_score = -1.0

    for name, model in model_candidates:
        scores = cross_val_score(model, X_train_vec, y, cv=cv, scoring='f1', groups=groups)
        score = float(scores.mean())
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    return best_name, best_model, round(best_score, 4)


def oof_probabilities(model, X_vec, y: pd.Series, groups: pd.Series):
    n_splits = get_group_cv_splits(y, groups)
    if n_splits < 2:
        return None

    cv = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=float)

    for tr_idx, val_idx in cv.split(X_vec, y, groups=groups):
        m = clone(model)
        m.fit(X_vec[tr_idx], y.iloc[tr_idx])
        oof[val_idx] = model_probability(m, X_vec[val_idx])

    return oof


def tune_threshold(y_true: pd.Series, probs: np.ndarray | None):
    if probs is None:
        return 0.5, None

    candidates = []
    for thr in np.arange(0.1, 0.91, 0.02):
        pred = (probs >= thr).astype(int)
        acc = float(accuracy_score(y_true, pred))
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
        candidates.append((float(thr), acc, float(f1)))

    # Accuracy floor avoids very low-precision thresholds while still prioritizing F1.
    best_acc_any = max(c[1] for c in candidates)
    acc_floor = max(0.55, best_acc_any - 0.08)
    eligible = [c for c in candidates if c[1] >= acc_floor]
    if not eligible:
        eligible = candidates

    best_threshold, best_acc, best_f1 = max(
        eligible,
        key=lambda x: (x[2], x[1], -abs(x[0] - 0.5))
    )

    return round(best_threshold, 2), {
        'oof_accuracy': round(float(best_acc), 4),
        'oof_f1': round(float(best_f1), 4),
        'selection_mode': 'maximize_f1_with_accuracy_floor',
        'accuracy_floor': round(float(acc_floor), 4),
    }


def default_dataset() -> pd.DataFrame:
    seed_rows = [
        {'text': 'If you do not act now, your family will suffer and society will collapse.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'Experts agree this is the only truth and questioning it is betrayal.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'This is your last chance to save yourself from disaster.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'You should compare multiple reports before making any decision.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'The policy has benefits and risks and should be debated publicly.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'Data from three independent surveys shows mixed public opinion.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'True believers must punish those who question our faith.', 'manipulative': 1, 'religious_manipulation': 1, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'God commands that only our religion should control the nation.', 'manipulative': 1, 'religious_manipulation': 1, 'political_manipulation': 1, 'anti_constitutional': 1},
        {'text': 'Religion is in danger and faithful people must fight now.', 'manipulative': 1, 'religious_manipulation': 1, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'All communities should enjoy equal rights under law.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'Interfaith dialogue can reduce conflict and strengthen trust.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'No religion should be used to threaten citizens.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'Only real patriots support this leader; critics are traitors.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 1, 'anti_constitutional': 0},
        {'text': 'The opposition lies and the media conspiracy hides the truth.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 1, 'anti_constitutional': 0},
        {'text': 'Vote for us or you will lose your freedom forever.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 1, 'anti_constitutional': 0},
        {'text': 'Citizens should evaluate manifestos and independent audits.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'The debate included representatives from all major parties.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'Election coverage presented both government and opposition views.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'We should suspend the constitution and cancel elections immediately.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 1, 'anti_constitutional': 1},
        {'text': 'Ban free speech and jail critics without trial to restore order.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 1, 'anti_constitutional': 1},
        {'text': 'There is no need for parliament; one ruler should decide everything.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 1, 'anti_constitutional': 1},
        {'text': 'Constitutional safeguards protect citizens from abuse of power.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'An independent judiciary is essential in a democracy.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'Civil liberties should be expanded, not reduced.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'Trusted leaders say this crisis proves only we can save the nation.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 1, 'anti_constitutional': 0},
        {'text': 'Experts proved that dissent is dangerous and should be silenced.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 1, 'anti_constitutional': 1},
        {'text': 'If you care about your children, obey now and do not ask questions.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'Our holy duty is to eliminate all sinful people from public life.', 'manipulative': 1, 'religious_manipulation': 1, 'political_manipulation': 1, 'anti_constitutional': 1},
        {'text': 'A balanced policy paper discussed trade-offs and limitations openly.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'The article cited sources and included counterarguments from critics.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'Community leaders encouraged peaceful participation in democratic institutions.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'Only our party can save culture, and opponents must be removed from office by force.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 1, 'anti_constitutional': 1},
        {'text': 'The constitution is weak and should be ignored in emergencies forever.', 'manipulative': 1, 'religious_manipulation': 0, 'political_manipulation': 1, 'anti_constitutional': 1},
        {'text': 'Citizens are encouraged to verify claims with multiple primary sources.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
        {'text': 'Public discussion should include minority voices and legal experts.', 'manipulative': 0, 'religious_manipulation': 0, 'political_manipulation': 0, 'anti_constitutional': 0},
    ]

    for i, row in enumerate(seed_rows):
        row['source_id'] = i

    def augment_text(text: str, idx: int) -> str:
        rng = random.Random(42 + idx)
        t = text

        replacements = [
            (' must ', ' should '),
            (' should ', ' ought to '),
            (' now', ' immediately'),
            (' immediately', ' right away'),
            (' citizens', ' people'),
            (' leader', ' leadership'),
            (' constitution', ' constitutional framework'),
            (' media', ' press'),
        ]
        rng.shuffle(replacements)
        for a, b in replacements[:2]:
            t = t.replace(a, b)

        prefixes = [
            'Breaking update: ',
            'Urgent bulletin: ',
            'Public message: ',
            'Political briefing: ',
            'Religious appeal: ',
            'Widely shared claim: ',
        ]
        suffixes = [
            ' This message is spreading quickly online.',
            ' Several posts repeat this framing.',
            ' The statement is being circulated repeatedly.',
            ' Supporters are amplifying this wording.',
            ' Critics say this framing is manipulative.',
            ' The claim is being framed as an emergency.',
        ]

        if idx % 2 == 0:
            t = rng.choice(prefixes) + t
        if idx % 3 == 0:
            t = t + rng.choice(suffixes)
        if idx % 5 == 0:
            t = t.replace(';', ', and')
        return ' '.join(t.split())

    target_size = 200
    rows = list(seed_rows)
    seen = {r['text'].strip().lower() for r in rows}

    idx = 0
    while len(rows) < target_size:
        base = seed_rows[idx % len(seed_rows)]
        candidate = dict(base)
        candidate['text'] = augment_text(base['text'], idx)
        key = candidate['text'].strip().lower()
        if key not in seen:
            seen.add(key)
            rows.append(candidate)
        idx += 1

    return pd.DataFrame(rows[:target_size])


def grouped_cv_summary(df: pd.DataFrame) -> dict:
    groups = df['source_id']
    n_splits = min(5, int(groups.nunique()))
    if n_splits < 2:
        return {
            'folds': 0,
            'macro_accuracy_mean': None,
            'macro_accuracy_std': None,
            'macro_f1_mean': None,
            'macro_f1_std': None,
        }

    cv = GroupKFold(n_splits=n_splits)
    fold_macro_acc = []
    fold_macro_f1 = []

    for train_idx, test_idx in cv.split(df['text'], groups=groups):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        vectorizer = build_vectorizer()
        X_train_vec = vectorizer.fit_transform(train_df['text'])
        X_test_vec = vectorizer.transform(test_df['text'])

        label_acc = []
        label_f1 = []
        train_groups = train_df['source_id']

        for label in LABELS:
            _, model, _ = choose_model(X_train_vec, train_df[label], train_groups)
            model.fit(X_train_vec, train_df[label])

            oof_probs = oof_probabilities(model, X_train_vec, train_df[label], train_groups)
            threshold, _ = tune_threshold(train_df[label], oof_probs)

            test_probs = model_probability(model, X_test_vec)
            test_pred = (test_probs >= threshold).astype(int)

            acc = accuracy_score(test_df[label], test_pred)
            _, _, f1, _ = precision_recall_fscore_support(
                test_df[label], test_pred, average='binary', zero_division=0
            )
            label_acc.append(float(acc))
            label_f1.append(float(f1))

        fold_macro_acc.append(float(np.mean(label_acc)))
        fold_macro_f1.append(float(np.mean(label_f1)))

    return {
        'folds': int(n_splits),
        'macro_accuracy_mean': round(float(np.mean(fold_macro_acc)), 4),
        'macro_accuracy_std': round(float(np.std(fold_macro_acc)), 4),
        'macro_f1_mean': round(float(np.mean(fold_macro_f1)), 4),
        'macro_f1_std': round(float(np.std(fold_macro_f1)), 4),
    }


def train_and_evaluate(output_dir: str = 'models') -> dict:
    df = default_dataset()
    os.makedirs(output_dir, exist_ok=True)

    # Prevent leakage: all augmented variants from the same seed stay in one split.
    groups = df['source_id']
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df['text'], groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df['text']
    X_test = test_df['text']
    y_train = train_df[LABELS]
    y_test = test_df[LABELS]
    train_groups = train_df['source_id']

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifiers = {}
    tuned_thresholds = {}
    metrics = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'sample_count': int(len(df)),
        'unique_source_count': int(df['source_id'].nunique()),
        'split_strategy': 'GroupShuffleSplit(source_id)',
        'vectorizer': 'FeatureUnion(word_tfidf + char_wb_tfidf)',
        'label_metrics': {},
    }

    macro_f1 = []
    macro_acc = []

    for label in LABELS:
        best_name, clf, cv_f1 = choose_model(X_train_vec, y_train[label], train_groups)
        clf.fit(X_train_vec, y_train[label])

        train_probs = model_probability(clf, X_train_vec)
        test_probs = model_probability(clf, X_test_vec)

        oof_probs = oof_probabilities(clf, X_train_vec, y_train[label], train_groups)
        threshold, threshold_quality = tune_threshold(y_train[label], oof_probs)

        pred_train = (train_probs >= threshold).astype(int)
        pred_test = (test_probs >= threshold).astype(int)

        acc = accuracy_score(y_test[label], pred_test)
        train_acc = accuracy_score(y_train[label], pred_train)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test[label], pred_test, average='binary', zero_division=0
        )

        classifiers[label] = clf
        tuned_thresholds[label] = threshold
        metrics['label_metrics'][label] = {
            'selected_model': best_name,
            'cv_f1': cv_f1,
            'tuned_threshold': threshold,
            'threshold_quality': threshold_quality,
            'accuracy': round(float(acc), 4),
            'train_accuracy': round(float(train_acc), 4),
            'accuracy_gap': round(float(train_acc - acc), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(float(f1), 4),
        }
        macro_f1.append(f1)
        macro_acc.append(acc)

    metrics['tuned_thresholds'] = tuned_thresholds
    metrics['overall'] = {
        'macro_accuracy': round(float(sum(macro_acc) / len(macro_acc)), 4),
        'macro_f1': round(float(sum(macro_f1) / len(macro_f1)), 4),
    }
    metrics['grouped_5fold_summary'] = grouped_cv_summary(df)

    joblib.dump(vectorizer, f'{output_dir}/vectorizer.joblib')
    joblib.dump(classifiers, f'{output_dir}/classifiers.joblib')

    with open(f'{output_dir}/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == '__main__':
    out = train_and_evaluate('models')
    print(json.dumps(out, indent=2))
