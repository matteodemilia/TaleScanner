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
    if "subordinateClauses" in selected_analysis:
        results["subordinateClauses"] = suborindate_clauses(text)
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
        totalsubordinate=results.get("subordinateClauses"),
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

    if totalCount == 0:
        return 0

    type_token_ratio = round((uniqueCount / totalCount), 2)
    return type_token_ratio

# finds the root token of a sentence, usually the main verb
# in instances there is a dependent clause, it is the verb of the independent clause
def find_root_of_sentence(doc):
    for token in doc:
        if token.dep_ == "ROOT":
            return token
        
    return None

# find the other verbs in the sentence
def find_other_verbs(doc, root_token):
    other_verbs = []

    if root_token is None:
        return other_verbs
    
    for token in doc:
        if token.pos_ == "VERB" and token != root_token:
            if token.dep_ in ["acl", "advcl", "relcl", "ccomp", "xcomp"]:
                other_verbs.append(token)
            else:
                ancestors = list(token.ancestors)
                if len(ancestors) == 1 and ancestors[0] == root_token:
                    other_verbs.append(token)
    return other_verbs

# find the token spans for each verb
def get_clause_token_span_for_verb(verb, doc, all_verbs):
    # Check if this is the only root token (main verb)
    if len(all_verbs) == 1:
        first_token_index = doc[0].i  # Initialize with the first token's index
        last_token_index = doc[-1].i
        for token in doc:
            if token.pos_ in ["PUNCT"]:
                last_token_index = token.i
                break
            last_token_index = token.i
        return (first_token_index, last_token_index + 1)

    else:
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

            elif token.dep_ in ["mark", "advcl", "relcl", "ccomp"]:
                if token.i < verb.i:
                    first_token_index = token.i
                else:
                    last_token_index = token.i

        return (first_token_index, last_token_index + 1)

# REQUIREMENT 5 - Number of clauses
@app.route("/num_clauses", methods=["POST"])
def num_clauses(text):
    # empty string
    if(text == " "):
        return 0;

    else:
        doc = nlp(text)
        all_clauses = []

        for s in doc.sents:
            root_token = find_root_of_sentence(s)
            if root_token is None:
                continue

            other_verbs = find_other_verbs(s, root_token)
            all_verbs = [root_token] + other_verbs

            token_spans = []
            for verb in all_verbs:
                token_span = get_clause_token_span_for_verb(verb, s, all_verbs)
                token_spans.append(token_span)

            sentence_clauses = []
            for start, end in token_spans:
                clause = s[start:end].text
                sentence_clauses.append(clause)

            all_clauses.extend(sentence_clauses)
   

    print(f"clauses: {all_clauses}")
    return len(all_clauses)

# REQUIREMENT 6 - Number of subordinate/dependent clauses
@app.route("/num_clauses", methods=["POST"])
def suborindate_clauses(text):
    doc = nlp(text)
    
    subordinate_clauses = 0
    current_clause = []
    subordinate_clauses_list = []
    in_subordinate_clause = False

    for sent in doc.sents:
        for token in sent:
            #print(token.dep_)
            if token.dep_ == "mark":  # because, that, if
                in_subordinate_clause = True
                subordinate_clauses += 1
                current_clause = [token.text]
            elif token.dep_ == "relcl":  # relative clause
                in_subordinate_clause = True
            elif token.dep_ == "advcl":  # adverbial clause
                in_subordinate_clause = True
            elif token.dep_ == "ccomp":  # clausal complement
                in_subordinate_clause = True
            elif token.dep_ == "ROOT" and in_subordinate_clause:
                in_subordinate_clause = False
                subordinate_clauses_list.append(" ".join(current_clause))  # Append the current clause to the list
                current_clause = []  # Reset the current clause

         # Check if there is a remaining subordinate clause at the end of the sentence
        if in_subordinate_clause:
            subordinate_clauses_list.append(" ".join(current_clause))
            current_clause = []
            in_subordinate_clause = False       

    print(f"subordinate clauses: {subordinate_clauses_list}")
    return subordinate_clauses


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
