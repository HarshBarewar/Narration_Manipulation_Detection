import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np


LABELS = [
    'manipulative',
    'religious_manipulation',
    'political_manipulation',
    'anti_constitutional',
]

DISPLAY_NAMES = {
    'manipulative': 'Manipulative',
    'religious_manipulation': 'Religious Manipulation',
    'political_manipulation': 'Political Manipulation',
    'anti_constitutional': 'Anti-Constitutional',
}

THRESHOLDS = {
    'manipulative': 0.55,
    'religious_manipulation': 0.52,
    'political_manipulation': 0.52,
    'anti_constitutional': 0.5,
}

RULE_KEYWORDS = {
    'manipulative': [
        'last chance', 'act now', 'if you do not', 'if you don\'t', 'everyone knows',
        'only truth', 'wake up', 'destroyed', 'collapse', 'urgent', 'must obey',
        'they hide', 'betrayal', 'panic', 'catastrophe', 'threat', 'fear', 'outrage',
        'survival', 'save your family', 'protect your family', 'before it is too late',
        'no other option', 'do not question', 'stop questioning', 'obey now',
        'hidden agenda', 'they are lying', 'brainwashed', 'you are in danger',
        'national crisis', 'crisis', 'enemy within', 'stand with us now',
        'must unite now', 'this is the only way', 'true patriots must',
        'silence critics', 'people must follow', 'you have no choice',
        'act before midnight', 'share this with everyone', 'mainstream media won\'t tell you',
        'they don\'t want you to know', 'wake people up', 'urgent warning',
        'final warning', 'irreversible damage', 'nothing will be left',
        'your children are at risk', 'if you stay silent', 'choose a side now',
        'do it now', 'do not wait', 'forced to obey', 'questioning is dangerous',
        'only we can save you', 'follow without doubt', 'believe this now',
        'time is running out', 'disaster is coming', 'they are covering it up',
        'secret plan', 'betray your nation', 'danger is everywhere',
        'you will regret it forever', 'obey before it is too late',
    ],
    'religious_manipulation': [
        'against god', 'true believers', 'holy duty', 'sacred duty',
        'divine punishment', 'faithful must', 'religion is in danger',
        'faith is in danger', 'blasphemy', 'sacred war', 'infidel',
        'sinful people', 'only our religion', 'only true faith', 'true faith',
        'god commands', 'will of believers', 'will of the believers',
        'enemies of the faith', 'enemy of faith', 'anti-faith', 'holy teachings',
        'religious authority', 'spiritual leaders', 'anger the divine',
        'defend our sacred values', 'sacred values', 'betray the faith',
        'against our religion', 'religious duty', 'follow the faith',
        'chosen by god', 'divine order', 'holy law above all', 'faithful warriors',
        'sacred command', 'god will punish dissenters', 'heaven demands obedience',
        'sin against god', 'curse on doubters', 'nonbelievers must submit',
        'purify society', 'cleanse the sinners', 'defend the holy path',
        'religious enemies', 'attack on our faith', 'offend the sacred',
        'faith under attack', 'holy mission', 'divine justice now',
        'stand for the faith now', 'betrayal of the holy book',
        'religious identity is threatened', 'serve god by force',
        'faithful citizens must act', 'god chooses our rulers',
        'questioning faith is betrayal', 'holy truth only',
    ],
    'political_manipulation': [
        'traitor', 'traitors', 'anti-national', 'real patriots', 'enemy party',
        'corrupt elite', 'media conspiracy', 'stolen election',
        'vote or lose freedom', 'national emergency', 'opposition lies',
        'only leader', 'silence dissent', 'remove opponents', 'enemy of nation',
        'enemies of the nation', 'nation is under attack', 'government must',
        'replace the constitution', 'remove constitutional protections',
        'absolute power', 'give absolute power', 'ban criticism', 'not allowed to vote',
        'should not be allowed to vote', 'should not be allowed to speak',
        'remove voting rights', 'remove opposition voices', 'single leader',
        'unite behind one leader', 'one nation one leader', 'opposition is enemy',
        'dissent is treason', 'critics are traitors', 'vote for us or perish',
        'the nation will fall', 'save the nation now', 'patriots only',
        'silence the opposition', 'ban opposition rallies',
        'remove anti-national voices', 'suspend political rights',
        'enemy agents within', 'the press is enemy', 'only our party is legitimate',
        'loyal citizens must obey', 'questioning leadership is betrayal',
        'punish dissenters', 'stop anti-government speech', 'cancel opposition',
        'our movement alone can save', 'one party rule', 'rule without opposition',
        'remove minority parties', 'purge political enemies',
        'anyone against us is against the nation',
    ],
    'anti_constitutional': [
        'suspend constitution', 'ignore the constitution', 'replace the constitution',
        'ban free speech', 'remove voting rights', 'abolish judiciary',
        'cancel elections', 'rule by force', 'jail critics without trial',
        'remove civil liberties', 'no need for parliament', 'end judicial review',
        'constitutional protections', 'remove constitutional protections',
        'ban criticism', 'not allowed to speak', 'silence criticism',
        'silence critics', 'absolute power to leaders', 'absolute power',
        'laws above constitution', 'replace constitutional law', 'undermine democracy',
        'end free expression', 'suppress dissent', 'no freedom of speech',
        'end fundamental rights', 'strip rights', 'no right to vote',
        'replace democratic institutions', 'indefinite emergency rule',
        'military rule now', 'rule by decree', 'shut down parliament',
        'courts must be dissolved', 'leaders above the law',
        'remove due process', 'detain without trial', 'ban public protest',
        'suspend civil rights', 'abolish elections', 'cancel voting permanently',
        'constitution is obsolete', 'constitution is irrelevant',
        'centralize all power', 'remove checks and balances',
        'judges should obey rulers', 'criminalize criticism', 'state controls speech',
        'end opposition rights', 'ban independent media', 'no independent judiciary',
        'dissolve democratic institutions', 'govern without legal limits',
        'executive above constitution', 'erase constitutional safeguards',
        'replace rights with state orders',
    ],
}


@dataclass
class DetectorArtifacts:
    vectorizer: object | None
    classifiers: Dict[str, object]
    metrics: Dict


class ManipulationDetector:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.artifacts = self._load_artifacts()

    def _load_artifacts(self) -> DetectorArtifacts:
        vectorizer_path = os.path.join(self.model_dir, 'vectorizer.joblib')
        classifiers_path = os.path.join(self.model_dir, 'classifiers.joblib')
        metrics_path = os.path.join(self.model_dir, 'metrics.json')

        vectorizer = None
        classifiers: Dict[str, object] = {}
        metrics: Dict = {
            'available': False,
            'message': 'Model metrics not available yet. Run `python train_model.py`.'
        }

        if os.path.exists(vectorizer_path) and os.path.exists(classifiers_path):
            vectorizer = joblib.load(vectorizer_path)
            loaded = joblib.load(classifiers_path)
            if isinstance(loaded, dict):
                classifiers = loaded

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                metrics['available'] = True

        return DetectorArtifacts(vectorizer=vectorizer, classifiers=classifiers, metrics=metrics)

    def get_metrics(self) -> Dict:
        return self.artifacts.metrics

    def _threshold_for(self, label: str) -> float:
        tuned = self.artifacts.metrics.get('tuned_thresholds', {})
        if isinstance(tuned, dict) and label in tuned:
            return float(tuned[label])
        return float(THRESHOLDS[label])

    @staticmethod
    def _sentence_split(article: str) -> List[str]:
        lines = [ln.strip() for ln in article.splitlines() if ln.strip()]
        if len(lines) > 1:
            return lines
        sentences = re.split(r'(?<=[.!?])\s+', article.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _safe_prob(model, x_vec) -> float:
        if hasattr(model, 'predict_proba'):
            return float(model.predict_proba(x_vec)[0][1])
        if hasattr(model, 'decision_function'):
            score = float(model.decision_function(x_vec)[0])
            return float(1.0 / (1.0 + np.exp(-score)))
        pred = int(model.predict(x_vec)[0])
        return float(pred)

    @staticmethod
    def _rule_score(text: str, label: str) -> Tuple[float, List[str]]:
        lowered = text.lower()
        matched = [kw for kw in RULE_KEYWORDS[label] if kw in lowered]
        if not matched:
            return 0.0, []
        score = min(0.95, 0.25 + 0.15 * len(matched))
        return score, matched

    def _predict_line(self, line: str) -> Dict[str, Dict]:
        outputs = {}

        if self.artifacts.vectorizer is not None and self.artifacts.classifiers:
            x_vec = self.artifacts.vectorizer.transform([line])
        else:
            x_vec = None

        for label in LABELS:
            ml_prob = 0.0
            if x_vec is not None and label in self.artifacts.classifiers:
                ml_prob = self._safe_prob(self.artifacts.classifiers[label], x_vec)

            rule_prob, matched = self._rule_score(line, label)
            combined_prob = ml_prob * 0.8 + rule_prob * 0.2 if x_vec is not None else rule_prob

            outputs[label] = {
                'ml_probability': round(ml_prob, 4),
                'rule_probability': round(rule_prob, 4),
                'probability': round(combined_prob, 4),
                'matched_terms': matched,
            }

        return outputs

    def analyze_article(self, article: str) -> Dict:
        segments = self._sentence_split(article)
        if not segments:
            return {'error': 'No valid text found.'}

        per_line = []
        for idx, seg in enumerate(segments, start=1):
            scores = self._predict_line(seg)
            per_line.append({'line_number': idx, 'text': seg, 'scores': scores})

        categories = {}
        for label in LABELS:
            line_probs = [entry['scores'][label]['probability'] for entry in per_line]
            avg_prob = float(np.mean(line_probs))
            max_prob = float(np.max(line_probs))
            final_score = round((avg_prob * 0.4) + (max_prob * 0.6), 4)
            threshold = self._threshold_for(label)

            evidence = []
            for entry in per_line:
                s = entry['scores'][label]
                if s['probability'] >= threshold or s['matched_terms']:
                    evidence.append({
                        'line_number': entry['line_number'],
                        'text': entry['text'],
                        'score': s['probability'],
                        'matched_terms': s['matched_terms'],
                    })

            categories[label] = {
                'display_name': DISPLAY_NAMES[label],
                'flagged': final_score >= threshold or len(evidence) > 0,
                'score': final_score,
                'threshold': threshold,
                'evidence_lines': evidence[:5],
            }

        risk_score = categories['manipulative']['score']
        if risk_score >= 0.7:
            risk_level = 'High'
        elif risk_score >= 0.45:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        return {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 4),
            'total_segments': len(segments),
            'categories': categories,
        }
