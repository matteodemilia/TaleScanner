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

# define flask Homepage when app runs
@app.route("/")
def index():
    return render_template("homepage.html")

# define flask results page
@app.route("/resultspage.html")
def results():
    return render_template("resultspage.html")

# define flask about page
@app.route("/aboutpage")
def about():
    return render_template("aboutpage.html")

# Gets input from homepage checkbox
@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    text = request.form["text"]
    selected_analysis = request.form.getlist("analysis")

    # declare results list to hold chosen results 
    results = {}

    # Check if HTML element id is chosen and sent to results list above, call functions 
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
    for token in doc:
        if token.dep_ == "ROOT":
            return token

# find the other verbs in the sentence
def find_other_verbs(doc, root_token):
    other_verbs = []
    for token in doc:
        if token.pos_ == "VERB" and token != root_token:
            ancestors = list(token.ancestors)
            if len(ancestors) == 1 and ancestors[0] == root_token:
                other_verbs.append(token)
    return other_verbs

# find the token spans for each verb
def get_clause_token_span_for_verb(verb, doc, all_verbs):
    first_token_index = verb.i
    last_token_index = verb.i

    for token in doc:
        if token in all_verbs:
            continue

        if token in list(verb.children):
            if token.i < first_token_index:
                first_token_index = token.i
            if token.i > last_token_index:
                last_token_index = token.i

        elif token.pos_ in ["CCONJ", "PUNCT"]:
            if token.i < verb.i:
                first_token_index = token.i + 1
            else:
                last_token_index = token.i
                break

    return (first_token_index, last_token_index + 1)

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

        '''
        # taken from https://subscription.packtpub.com/book/data/9781838987312/2/ch02lvl1sec13/splitting-sentences-into-clauses
        doc = nlp(text)

        # calls function 
        root_token = find_root_of_sentence(doc)
        print(f"root_token: {root_token}")

        # use preceding function to find the remaining
        other_verbs = find_other_verbs(doc, root_token)
        print(f"other verbs: {other_verbs}")

        # put together all the verbs in one array
        token_spans = []   
        all_verbs = [root_token] + other_verbs
        for other_verb in all_verbs:
            (first_token_index, last_token_index) = \
            get_clause_token_span_for_verb(other_verb, 
                                            doc, all_verbs)
            token_spans.append((first_token_index, 
                                last_token_index))
            
        print(f"token_spans: {token_spans}")
            
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
        '''
        all_clauses = []
        sentences = [sent.text for sent in nlp(text).sents]

        for sentence in sentences:
            doc = nlp(sentence)
            root_token = find_root_of_sentence(doc)
            other_verbs = find_other_verbs(doc, root_token)
            all_verbs = [root_token] + other_verbs

            token_spans = []
            for verb in all_verbs:
                token_span = get_clause_token_span_for_verb(verb, doc, all_verbs)
                token_spans.append(token_span)

            sentence_clauses = []
            for token_span in token_spans:
                start, end = token_span
                clause = doc[start:end].text
                sentence_clauses.append(clause)

            all_clauses.extend(sentence_clauses)

    #print(f"num clauses: {len(all_clauses)}")
    return len(all_clauses)

# REQUIREMENT 8 - Verb errors
@app.route("/verbErr", methods=["POST"])
def verbEs(texts):
    doc = nlp(texts)
    counter = 0
    assert doc.has_annotation("SENT_START")
    for sent in doc.sents:
        # print(sent.text)
        sent1 = sent.text

        corrected_sentences = gf.correct(sent1, max_candidates=1)

        # print("[Input] ", sent1)
        test = str(corrected_sentences)
        for corrected_sentence in corrected_sentences:
            counter = counter + 1
            hold = gf.get_edits(sent1, corrected_sentence)
            if hold == []:
                # print("no change")
                counter = counter - 1
        # print("-" * 100)
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
        if (token.is_punct == True):
            print("should not be counted")
            pass
        else:
            if plur == ["Plur"]:
                counter = counter + 1
            if tense == ["Past"]:
                counter = counter + 1
            counter = counter + 1
        print(counter)
    return counter


if __name__ == "__main__":
    app.run(debug=False)
