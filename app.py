from flask import Flask, request, jsonify, render_template
from nltk.tokenize import sent_tokenize, word_tokenize
from spellchecker import SpellChecker

spell = SpellChecker()
from main import (
    spellCheckForWeb,
    nltk_data,
    bulidDictionarty,
    buidBigram,
    should_skip,
    clean_token,
    CONFUSION_SET
)

def hunspellCheck(text):
    results = []
    sentences = sent_tokenize(text)
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        for token in tokens:
            result = clean_token(token)
            if result is None:
                continue
            org, clean = result
            if should_skip(org, clean, 0):
                continue
            if clean not in spell:
                suggestions = list(spell.candidates(clean))[:3]
                results.append({
                    'word':        org,
                    'type':        'unknown',
                    'suggestions': suggestions
                })
    return results


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'ContextSpell': [], 'hunspell': []})
    
    ContextSpell_results= spellCheckForWeb(text)
    hunspell_results = hunspellCheck(text)
    
    return jsonify({
        'ContextSpell':  ContextSpell_results,
        'hunspell': hunspell_results
    })


if __name__ == '__main__':
    nltk_data()
    
    import main
    main.dictionary, main.dictionary_by_letter = bulidDictionarty()
    main.bigramCounts = buidBigram()
    
    app.run(debug=True)


