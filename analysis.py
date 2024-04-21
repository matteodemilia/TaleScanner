import spacy
from flask import Flask, request, session, render_template
from gramformer import Gramformer
import torch
import os
# from morphemes import Morphemes

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1212)  # used for gramformer

gf = Gramformer(models=1, use_gpu=False)  # 1=corrector, 2 = detector(WIP)

nlp = spacy.load("en_core_web_trf")
app = Flask(__name__, static_folder="static")
app.secret_key = os.urandom(24) # to generate a session ID
path = "./data"

app.config['FLASK_DEBUG'] = 0

# define flask Homepage when app runs
@app.route("/")
def index():
    selected_analysis = session.get("selected_analysis", [])
    return render_template("homepage.html", selected_analysis=selected_analysis)

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
    session["selected_analysis"] = selected_analysis

    # declare results list to hold chosen results 
    results = {}
    # verb_errors = 0
    clauses = 0

    # Check if HTML element id is chosen and sent to results list above, call functions 
    if "totalWords" in selected_analysis:
        words, first, last = total_words(text)
        results["totalWords"] = {"total": words, "firstword": first, "lastword": last}
    if "differentWords" in selected_analysis:
        different_words_count, different_words_list = different_words(text)
        results["differentWords"] = {"count": different_words_count, "list": different_words_list}
    if "typeToken" in selected_analysis:
        ttr, unique, total = type_token_ratio(text)
        results["typeToken"] = {"typetokenratio": ttr, "differentwords": unique, "totalwords": total}
    if "subordinateClauses" in selected_analysis:
        results["subordinateClauses"] = suborindate_clauses(text)
    if "totalClauses" in selected_analysis:
        clauses =  num_clauses(text)
        results["totalClauses"] = num_clauses(text)
    if "syntacticSubordination" in selected_analysis:
        ssindex, sub, numclauses = syntactic_subordination_index(text)
        results["syntacticSubordination"] = {"index": ssindex, "subordinateClauses": sub, "totalClauses": numclauses}
    if "morpheme" in selected_analysis:
        results["morpheme"] = morph(text)
    if "verbErr" in selected_analysis:
        error_count, verb_errors =  verbEs(text)
        results["verbErr"] = {"count": error_count, "list": verb_errors}

    # if "verbClauses" and "verbErr" in selected_analysis:
    #     ans, ve, cl = verb_clauses(error_count, clauses) # passing verb+clauses to avoid redundancy
    #     results["verbClauses"] = {"verbClauses": ans, "verbErrors": ve, "totalClauses": cl}
    if "verbClauses" in selected_analysis:
        error_count, verb_errors =  verbEs(text) 
        clauses =  num_clauses(text)
        ans, ve, cl = verb_clauses(error_count, clauses) 
        results["verbClauses"] = {"verbClauses": ans, "verbErrors": error_count, "totalClauses": clauses}
    if "wordsClauses" in selected_analysis:
        ans, w, c = words_per_clause(text)
        results["wordsClauses"] = {"wordsPerClause": ans, "totalWords": w, "totalClauses": c}



    return render_template(
        "resultspage.html",
        results=results,
        text=text,
        selected_analysis=selected_analysis,
        # storing results into new variables so i can style each analysis on results page here
        totalwords=results.get("totalWords"),
        differentwords=results.get("differentWords"),
        typetoken=results.get("typeToken"),
        totalsubordinate=results.get("subordinateClauses"),
        totalclauses=results.get("totalClauses"),
        syntacticsubordination=results.get("syntacticSubordination"),
        morphemes=results.get("morpheme"),
        verberrors=results.get("verbErr"),
        verbclauses=results.get("verbClauses"),
        wordsperclauses=results.get("wordsClauses")
    )

# REQUIREMENT 1 - Total number of words
@app.route("/total_words", methods=["POST"])
def total_words(text):
    if (len(text) == 0):
        return 0,0,0
    
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha]
    num_words = len(words)

    first = words[0]
    last = words[-1]

    return num_words, first, last


# REQUIREMENT 2 - Number of different words
@app.route("/different_words", methods=["POST"])
def different_words(text):
    doc = nlp(text)
    words = set([
        token.text.lower() for token in doc if token.is_alpha
    ])  # not case sensitive
    num_words = len(set(words))  # put in set, only unique elements
    return num_words, words


# REQUIREMENT 3 - unique words / total number of words
@app.route("/unique_words", methods=["POST"])
def type_token_ratio(text):
    doc = nlp(text)

    words = [token.text for token in doc if token.is_alpha]
    totalCount = len(words)

    words = [
        token.text.lower() for token in doc if token.is_alpha
    ] 
    uniqueCount = len(set(words))

    if totalCount == 0:
        return 0, 0, 0
 
    type_token_ratio = round((uniqueCount / totalCount), 2)
    return type_token_ratio, uniqueCount, totalCount

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
    if not text.strip():
        return 0

    doc = nlp(text)
    all_clauses = []

    for sent in doc.sents:
        root_token = find_root_of_sentence(sent)
        if root_token is None:
            continue

        other_verbs = find_other_verbs(sent, root_token)
        all_verbs = [root_token] + other_verbs

        clause_token_spans = []
        for verb in all_verbs:
            clause_token_span = get_clause_token_span_for_verb(verb, sent, all_verbs)
            clause_token_spans.append(clause_token_span)

        for start, end in clause_token_spans:
            clause = sent[start:end].text.strip()
            all_clauses.append(clause)

    # Combine clauses connected by conjunctions
    combined_clauses = []
    conj = ["while", "since", "whenever", "because", "although", "as"]
    current_clause = all_clauses[0]
    for i in range(1, len(all_clauses)): # if i in conj
        if i in conj:
            combined_clauses.append(current_clause)
            current_clause = all_clauses[i]
        else:
            current_clause += " " + all_clauses[i]
    combined_clauses.append(current_clause)
   
    print(f"clauses: {all_clauses}")
    return len(all_clauses)

# REQUIREMENT 6 - Number of subordinate/dependent clauses
@app.route("/subordinate_clauses", methods=["POST"])
def suborindate_clauses(text):
    doc = nlp(text)
    prep = ["despite", "because"] # words thats not working..
    
    subordinate_clauses = 0
    current_clause = []
    subordinate_clauses_list = []
    in_subordinate_clause = False

    for sent in doc.sents:
        for token in sent:
            #print(token.dep_)
            if token.text.lower() in prep: 
                in_subordinate_clause = True
                subordinate_clauses += 1
                current_clause = [token.text]
            elif token.dep_ == "mark":  # because, that, if
                in_subordinate_clause = True
                subordinate_clauses += 1
                current_clause = [token.text]
            elif token.dep_ in ["relcl", "advcl", "ccomp"]:  # relative clause, adverbial clause, clausal complement
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

# REQUIREMENT 7 - Syntactic Subordination Index ( Subordinate clauses/total clauses )
@app.route("/syntactic_subordination_index", methods=["POST"])
def syntactic_subordination_index(text):
    total_subordinate = suborindate_clauses(text)
    total_clauses = num_clauses(text)

    if total_clauses == 0:
        return 0, 0, 0
    
    index = round((total_subordinate/total_clauses),2)

    return index, total_subordinate, total_clauses

# REQUIREMENT 8 - Verb errors
@app.route("/verbErr", methods=["POST"])
def verbEs(texts):
    doc = nlp(texts)
    counter = 0
    bad_sentences = []
    assert doc.has_annotation("SENT_START")
    for sent in doc.sents:
        # print(sent.text)
        sent1 = sent.text
        #print(sent1)
        corrected_sentences = gf.correct(sent1, max_candidates=1)

        test = str(corrected_sentences)
        for corrected_sentence in corrected_sentences:
            hold = gf.get_edits(sent1, corrected_sentence)
            if hold == []:
                # print("no change")
                counter = counter - 1
            else:
                for data in hold:
                    altverb = data[1]
                    counter = counter + 1
            #print(altverb)
            
            bad_sentences.append(sent1)
        # print("-" * 100)
    return counter, bad_sentences

# Addional Requirement - Words per clause
@app.route("/words_per_clause", methods=["POST"])
def words_per_clause(text):
    words, f, l = total_words(text)
    clauses = num_clauses(text)

    if clauses == 0:
        return 0, 0, 0
    
    ans = round((words/clauses),2)

    return ans, words, clauses



# REQUIREMENT 9 - Verb errors divided by the number of clauses
@app.route("/verbClauses", methods=["POST"])
def verb_clauses(verbs, clauses):
    if clauses == 0:
        return 0, 0, 0

    ratio = round((verbs/clauses),2)

    return ratio, verbs, clauses

    
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
