import spacy
from morphemes import Morphemes
from flask import Flask, request, render_template
from gramformer import Gramformer
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1212) #used for gramformer

gf = Gramformer(models =1, use_gpu=False) #1=corrector, 2 = detector(WIP)


nlp = spacy.load('en_core_web_trf')
app = Flask(__name__, static_folder='static')
path = "./data"


# Homepage when app runs
@app.route('/')
def index():
    return render_template('homepage.html')

# Gets input from checkbox
@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.form['text']
    selected_analysis = request.form.getlist('analysis')

    results = {}

    if 'totalWords' in selected_analysis:
        results['totalWords'] = total_words(text)
    if 'differentWords' in selected_analysis:
        results['differentWords'] = different_words(text)
    if 'typeToken' in selected_analysis:
        results['typeToken'] = type_token_ratio(text)
    if 'morpheme' in selected_analysis:
        results['morpheme'] = morph(text)
    if 'verbErr' in selected_analysis:
        results['verbErr'] = verbEs(text)

    return render_template('homepage.html', results=results)

# REQUIREMENT 1 - Total number of words
@app.route('/total_words', methods=['POST'])
def total_words(text):
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha]
    num_words = len(words)
    return num_words

# REQUIREMENT 2 - Number of different words
@app.route('/different_words', methods=['POST'])
def different_words(text):
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha] # not case sensitive
    num_words = len(set(words)) # put in set, only unique elements
    return num_words

# REQUIREMENT 3 - unique words / total number of words
@app.route('/unique_words', methods=['POST'])
def type_token_ratio(text):
    doc = nlp(text)
    totalCount = total_words(text)
    uniqueCount = different_words(text)
    return round((uniqueCount/totalCount),2)

# REQUIREMENT - Verb errors
@app.route('/verbErr', methods=['POST'])
def verbEs(texts):
    doc = nlp(texts)
    counter = 0
    assert doc.has_annotation("SENT_START")
    for sent in doc.sents:
        print(sent.text)
        sent1 = sent.text
        
        corrected_sentences = gf.correct(sent1, max_candidates=1)
        
        print("[Input] ", sent1)
        test = str(corrected_sentences)
        for corrected_sentence in corrected_sentences:
            counter = counter + 1
            hold= gf.get_edits(sent1, corrected_sentence)
            if (hold == []):
                print("no change")
                counter = counter - 1
        print("-" *100)
    return(counter)

# REQUIREMENT - Morphemes
@app.route('/morpheme', methods=['POST'])
def morph(text):
    doc =nlp(text)
    counter = 0
    for token in doc:
        tense = token.morph.get('Tense')
        plur = token.morph.get('Number')
        print(token, tense, plur)
        if(plur == ['Plur']):
            counter = counter + 1
        if(tense == ['Past']):
            counter = counter + 1
        counter = counter + 1
    return counter
    
    ### when using morphemes library
    ### the complexity is too high and makes very slow
    #m=Morphemes(path)
    #c = 0
    #test = text.split()
    #for i in test:
    #    c = c+ (m.count(i))
    #    print(c)
    #   print(m.parse(i))
    #
    #return (c)
    
    
if __name__ == '__main__':
    app.run(debug=True)