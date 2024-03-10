import spacy

# from morphemes import Morphemes
from flask import Flask, request, render_template
from gramformer import Gramformer
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(1212)  # used for gramformer

gf = Gramformer(models=1, use_gpu=False)  # 1=corrector, 2 = detector(WIP)


nlp = spacy.load("en_core_web_trf")
app = Flask(__name__, static_folder="static")
path = "./data"

app.config['FLASK_DEBUG'] = 0

# Homepage when app runs
@app.route("/")
def index():
    return render_template("homepage.html")

# results page
@app.route("/resultspage.html")
def results():
    return render_template("resultspage.html")

# about page
@app.route("/aboutpage")
def about():
    return render_template("aboutpage.html")

# Gets input from homepage checkbox
@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    text = request.form["text"]
    selected_analysis = request.form.getlist("analysis")

    results = {}

    if "totalWords" in selected_analysis:
        results["totalWords"] = total_words(text)
    if "differentWords" in selected_analysis:
        results["differentWords"] = different_words(text)
    if "typeToken" in selected_analysis:
        results["typeToken"] = type_token_ratio(text)
    if "totalClauses" in selected_analysis:
        results["totalClauses"] = num_clauses(text)
    if "morpheme" in selected_analysis:
        results["morpheme"] = morph(text)
    if "verbErr" in selected_analysis:
        results["verbErr"] = verbEs(text)

    return render_template(
        "resultspage.html",
        results=results,
        text=text,
        # storing results into new variables so i can style each analysis on results page here
        totalwords=results.get("totalWords"),
        differentwords=results.get("differentWords"),
        typetoken=results.get("typeToken"),
        totalclauses=results.get("totalClauses"),
        morphemes=results.get("morpheme"),
        verberrors=results.get("verbErr"),
    )


# REQUIREMENT 1 - Total number of words
@app.route("/total_words", methods=["POST"])
def total_words(text):
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha]
    num_words = len(words)
    return num_words


# REQUIREMENT 2 - Number of different words
@app.route("/different_words", methods=["POST"])
def different_words(text):
    doc = nlp(text)
    words = [
        token.text.lower() for token in doc if token.is_alpha
    ]  # not case sensitive
    num_words = len(set(words))  # put in set, only unique elements
    return num_words


# REQUIREMENT 3 - unique words / total number of words
@app.route("/unique_words", methods=["POST"])
def type_token_ratio(text):
    doc = nlp(text)
    totalCount = total_words(text)
    uniqueCount = different_words(text)
    return round((uniqueCount / totalCount), 2)

# finds the root token of a sentence, usually the main verb
# in instances there is a dependent clause, it is the verb of the independent clause
def find_root_of_sentence(doc):
    root_token = None
    for token in doc:
        if (token.dep_ == "ROOT"):
            root_token = token
    return root_token

# find the other verbs in the sentence
def find_other_verbs(doc, root_token):
    other_verbs = []
    for token in doc:
        ancestors = list(token.ancestors)
        if (token.pos_ == "VERB" and len(ancestors) == 1\
            and ancestors[0] == root_token):
            other_verbs.append(token)
    return other_verbs

# find the token spans for each verb
def get_clause_token_span_for_verb(verb, doc, all_verbs):
    first_token_index = len(doc)
    last_token_index = 0
    this_verb_children = list(verb.children)
    for child in this_verb_children:
        if (child not in all_verbs):
            if (child.i < first_token_index):
                first_token_index = child.i
            if (child.i > last_token_index):
                last_token_index = child.i
    return(first_token_index, last_token_index)

# REQUIREMENT 5 - Number of clauses
@app.route("/num_clauses", methods=["POST"])
def num_clauses(text):
    # empty string
    if(text == " "):
        return 0;

    else:
        '''
        # original idea - keeping just in case
        doc = nlp(text)
        conjunctions = {'and', 'but', 'so', 'because', 'if', 'or', 'yet', 'nor', 'while', 
                        'although', 'though', 'since', 'unless', 'whereas', 'whether'}
        clauses = []
        current_clause = []
        
        for token in doc:
            if token.text in ['.', ';', ',', ':', '?', '!'] or (token.pos_ == 'CCONJ') or (token.text.lower() in conjunctions):
                if current_clause:
                    clauses.append(current_clause)
                    current_clause = []
            else:
                current_clause.append(token.text)

        if current_clause:
            clauses.append(current_clause)
        '''
    
    doc = nlp(text)

    # calls function
    root_token = find_root_of_sentence(doc)

    # use preceding function to find the remaining
    other_verbs = find_other_verbs(doc, root_token)

    # put together all the verbs in one array
    token_spans = []   
    all_verbs = [root_token] + other_verbs
    for other_verb in all_verbs:
        (first_token_index, last_token_index) = \
        get_clause_token_span_for_verb(other_verb, 
                                        doc, all_verbs)
        token_spans.append((first_token_index, 
                            last_token_index))
        
    sentence_clauses = []
    for token_span in token_spans:
        start = token_span[0]
        end = token_span[1]
        if (start < end):
            clause = doc[start:end]
            sentence_clauses.append(clause)
    sentence_clauses = sorted(sentence_clauses, 
                            key=lambda tup: tup[0])

    clauses_text = [clause.text for clause in sentence_clauses]
    print(f"clauses_text: {clauses_text}")

    return len(clauses_text)

# REQUIREMENT 8 - Verb errors
@app.route("/verbErr", methods=["POST"])
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
            hold = gf.get_edits(sent1, corrected_sentence)
            if hold == []:
                print("no change")
                counter = counter - 1
        print("-" * 100)
    return counter


# REQUIREMENT 4 - Morphemes
@app.route("/morpheme", methods=["POST"])
def morph(text):
    doc = nlp(text)
    counter = 0
    for token in doc:
        tense = token.morph.get("Tense")
        plur = token.morph.get("Number")
        print(token, tense, plur)
        if plur == ["Plur"]:
            counter = counter + 1
        if tense == ["Past"]:
            counter = counter + 1
        counter = counter + 1
    return counter

    ### when using morphemes library
    ### the complexity is too high and makes very slow
    # m=Morphemes(path)
    # c = 0
    # test = text.split()
    # for i in test:
    #    c = c+ (m.count(i))
    #    print(c)
    #   print(m.parse(i))
    #
    # return (c)


if __name__ == "__main__":
    app.run(debug=False)
