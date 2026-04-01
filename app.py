import os
from flask import Flask, jsonify, render_template, request

from detector import ManipulationDetector

app = Flask(__name__)
detector = ManipulationDetector()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze')
def analyze_page():
    return render_template('analyze.html')


@app.route('/results')
def results_page():
    return render_template('results.html')


@app.route('/accuracy')
def accuracy_page():
    return render_template('accuracy.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_api():
    payload = request.get_json(silent=True) or {}
    article = (payload.get('article') or '').strip()

    if not article:
        return jsonify({'error': 'Please provide article text.'}), 400

    result = detector.analyze_article(article)
    return jsonify(result)


@app.route('/api/metrics')
def metrics_api():
    return jsonify(detector.get_metrics())


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
