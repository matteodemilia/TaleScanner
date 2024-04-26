import spacy
from flask import Flask, request, session, render_template, url_for, flash
from gramformer import Gramformer
import torch
import os
from morphemes import Morphemes

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

# checks whether 1. no text and checkbox 2. no text 3. no checkbox
def validateResults(text, analysis):
    if not text.strip() and not analysis:
        error_message = "ERROR: Please enter text and check at least one checkbox."
        return render_template("homepage.html", error_message=error_message, analysis=analysis)
    if not text.strip():
        error_message = "ERROR: Please enter text"
        return render_template("homepage.html", error_message=error_message, analysis=analysis)
    elif not analysis:
        error_message = "ERROR: Please check at least one checkbox."
        return render_template("homepage.html", error_message=error_message, analysis=analysis)
    else:
        return None

# Gets input from homepage checkbox
@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    text = request.form["text"]
    selected_analysis = request.form.getlist("analysis")
    session["selected_analysis"] = selected_analysis

    result = validateResults(text, selected_analysis)

    if result is not None: # there are errors , else continue as normal
        return result

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
        num, l = subordinate_clauses(text)
        results["subordinateClauses"] = {"numSubordinate": num, "list": l}
    if "totalClauses" in selected_analysis:
        c, l =  num_clauses(text)
        results["totalClauses"] = {"numClauses": c, "list": l}
    if "syntacticSubordination" in selected_analysis:
        ssindex, sub, numclauses = syntactic_subordination_index(text)
        results["syntacticSubordination"] = {"index": ssindex, "subordinateClauses": sub, "totalClauses": numclauses}
    if "morpheme" in selected_analysis:
        counter, lemma, bound, free = morph(text)
        results["morpheme"] = {"count": counter, "list": lemma, "bound": bound, "free": free}
    if "verbErr" in selected_analysis:
        error_count, verb_errors =  verbEs(text)
        results["verbErr"] = {"count": error_count, "list": verb_errors}
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
    # commented out original idea for back up
    # if (len(text) == 0):
    #     return 0,0,0
    
    # doc = nlp(text)
    # words = [token.text for token in doc if token.is_alpha]
    # num_words = len(words)

    # first = words[0]
    # last = words[-1]

    # return num_words, first, last

    # NEW SOLUTION:
    # dashed words ex. "hello-hello-world" counts as 1
    # numbers ex. "123" or "343254" is counted as a word
    # hi&&hi counts as 1 in the mean time ["hihi"]

    words = []
    count = 0
    current_word = ""

    for char in text:
        if char.isalnum() or (char == '-') :
            current_word += char
        elif char == ' ':
            words.append(current_word)
            count += 1
            current_word = ""

    if current_word:
        words.append(current_word)
        count += 1

    first = words[0]
    last = words[-1]    

    return count, first, last


# REQUIREMENT 2 - Number of different words
@app.route("/different_words", methods=["POST"])
def different_words(text):
    # doc = nlp(text)
    # words = set([
    #     token.text.lower() for token in doc if token.is_alpha
    # ])  # not case sensitive
    # num_words = len(set(words))  # put in set, only unique elements
    # return num_words, words

    words = []
    count = 0
    current_word = ""

    for char in text:
        if char.isalnum() or (char == '-') :
            current_word += char
        elif char == ' ':
            words.append(current_word)
            count += 1
            current_word = ""

    if current_word:
        words.append(current_word)
        count += 1

    unique_words = set(words)

    return len(unique_words), unique_words


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
    conjunctions = ["if", "when", "because", "since", "although", "unless", "until"]  # words that wern't working

    for token in doc:
        if token.pos_ == "VERB" and token != root_token:
            if token.dep_ in ["acl", "advcl", "relcl", "ccomp", "xcomp"]:
                other_verbs.append(token)
            elif token.text.lower() in conjunctions:  
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
        if root_token is None or root_token.pos_ != "VERB":
            continue  # Skip if no root token or root token is not a verb

        other_verbs = find_other_verbs(sent, root_token)
        all_verbs = [root_token] + other_verbs

        # Identify main clause
        main_clause_span = get_clause_token_span_for_verb(root_token, sent, all_verbs)
        main_clause = sent[main_clause_span[0]:main_clause_span[1]].text.strip()
        all_clauses.append(main_clause)

        # Identify subordinate clauses
        for verb in other_verbs:
            clause_token_span = get_clause_token_span_for_verb(verb, sent, all_verbs)
            clause = sent[clause_token_span[0]:clause_token_span[1]].text.strip()
            all_clauses.append(clause)

    print(f"clauses: {all_clauses}")
    return len(all_clauses)

# REQUIREMENT 6 - Number of subordinate/dependent clauses
@app.route("/subordinate_clauses", methods=["POST"])
def subordinate_clauses(text):
    doc = nlp(text)
    # words thats not working..
    conjunctions = ["despite", "because", "that", "who", "when", "where", "what"] 

    # conjunctions = [
    # "and", "but", "or", "nor", "for", "so", "yet",
    # "although", "in order that", "whatever",
    # "as", "provided that", "when",
    # "as if", "since", "whenever",
    # "as long as", "so that", "where",
    # "because", "than", "whereas",
    # "before", "that", "wherever",
    # "even if", "though", "whether",
    # "even though", "unless",
    # "ever since", "until", "if"
    #]
    
    subordinate_clauses = 0
    current_clause = []
    subordinate_clauses_list = []
    in_subordinate_clause = False

    for sent in doc.sents:
        for token in sent:
            #print(token.dep_)
            if token.text.lower() in conjunctions: 
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
    return subordinate_clauses, subordinate_clauses_list

# REQUIREMENT 7 - Syntactic Subordination Index ( Subordinate clauses/total clauses )
@app.route("/syntactic_subordination_index", methods=["POST"])
def syntactic_subordination_index(text):
    total_subordinate, list = subordinate_clauses(text)
    total_clauses, list = num_clauses(text)

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
                pass
            else:
                for data in hold:
                    altverb = data[1]
                    counter = counter + 1
                    bad_sentences.append(sent1)
            #print(altverb)
            

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
    m = Morphemes(path)
    lemma = []
    free = []
    bound = []
    for token in doc:
        if (token.is_punct == True): #need to get rid of punctuations that mess up tokenization and morphemes library
            token = "bad input" #changes the token so the program stays happy
            pass #skip this token for parsing
        else:
            word = str(token)
            data = m.parse(word)
            #print("MORPH:   ", data)
            status = data['status']
            word = data['word']
            morpheme_count = data['morpheme_count']
            
            #print("Morpheme Count:", morpheme_count)
            if word.isalpha() == False:
                pass
            elif status != 'NOT_FOUND':
                tree = data['tree']
                if 'tree' in data:
                    tree = data['tree']                
                    for item in tree:
                        if 'children' in item:  # If it's a free morpheme
                            free_morpheme = item['children'][0]['text']
                            free.append(free_morpheme)
                        else:  # If it's a bound morpheme
                            bound_morpheme = item['text']
                            bound.append(bound_morpheme)
                else:
                    print("tree key not found in library")
                
            lemma.append(token.lemma_)
            tense = token.morph.get("Tense")
            plur = token.morph.get("Number")
            verbform = token.morph.get("VerbForm")
            print("token: ",token, "  pos: ",token.pos_, "  Morph: ", token.morph )
            if (token.is_punct == True):
                pass
            else:
                if plur == ["Plur"]:
                    counter = counter + 1
                elif tense == ["Past"]:
                    counter = counter + 1
                elif verbform == ["Part"]:
                    counter = counter + 1
                counter = counter + 1
            lemma = list(set(lemma)) #sends only unique lemmas, reduces mutliple of same word.
            free = list(set(free))
            bound = list(set(bound))
    return counter, lemma, bound, free


if __name__ == "__main__":
    app.run(debug=False)
