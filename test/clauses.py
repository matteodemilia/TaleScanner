import spacy
from flask import Flask, request, session, render_template, url_for, flash
from gramformer import Gramformer
import torch
import os
from morphemes import Morphemes
import re

nlp = spacy.load("en_core_web_sm")

def subordindate_clauses(text):
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
    return subordinate_clauses

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
    sent_start = verb.sent.start
    sent_end = verb.sent.end

    for token in doc[sent_start:sent_end]:
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

    return (sent_start + first_token_index, sent_start + last_token_index + 1)

# REQUIREMENT 5 - Number of clauses
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
   
    print(f"clauses: {all_clauses}")
    return len(all_clauses)

while True:
    text = input("Enter text (or type 'end' to exit): ")
    if text.lower() == 'end':
        break
    #print(subordindate_clauses(text))
    print(num_clauses(text))