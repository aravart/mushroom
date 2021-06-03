from collections import deque
from nltk.translate.bleu_score import sentence_bleu
from subprocess import call
import io
import os
import numpy as np
import tqdm
import re
import argparse
import logging
import re
import subprocess
import sys
import nltk
import csv
import tempfile

if sys.version_info <= (3, 0):
     print("Please use Python 3. This script does not perform correctly on Python 2.")
     sys.exit(0)

epsilon = 0.01

parser = argparse.ArgumentParser()
parser.add_argument("filename")
# TODO pop open editor to add initial split, one | for indiv, two for sub
# Open up editor
parser.add_argument("-d", "--debug", action='store_true')
parser.add_argument("--depth", default=2, type=int)
parser.add_argument("--output", default=None)
parser.add_argument("--matcher", default='regex', choices=['regex','parse'])
parser.add_argument("--debug-seed-files", default=None)

logging.basicConfig(level=logging.WARN, format='%(message)s')

def load_corpus(filename):
     with open(filename) as csvfile:
          reader = csv.reader(csvfile, skipinitialspace=True)
          return [' '.join(nltk.word_tokenize(row[0])) for row in reader]

def mk_graph(seed_keyword_phrases, matcher, max_depth):
    h = deque([(keyword_phrase, 1) for keyword_phrase in seed_keyword_phrases])
    closed = set()
    g = dict([(keyword_phrase, {}) for keyword_phrase in seed_keyword_phrases])
    left = set()
    right = set(seed_keyword_phrases)

    while h:
        u, depth = h.popleft()
        if u in closed:
            continue
        else:
            closed.add(u)
        context = '__' in u
        if context:
            logging.info('pop (context): ' + u)
            matches = matcher.context_to_keyphrases(u)
        else:
            logging.info('pop (keyword): ' + u)
            matches = matcher.keyphrase_to_contexts(u)
        for v in matches:
            logging.info('  match: ' + v)
            if v not in g:
                g[v] = {}
                if context:
                    right.add(v)
                else:
                    left.add(v)
            if context:
                t = matcher.combine(u,v)
            else:
                t = matcher.combine(v,u)
            g[u][v] = epsilon * matcher.counts[t]
            g[v][u] = epsilon * matcher.counts[t]
            if depth < max_depth:
                h.append((v,depth + 1))
        if context:
             matches = matcher.context_to_contexts(u)
        else:
             matches = matcher.keyphrase_to_keyphrases(u)
        for v,w in matches:
            if w == 0:
                 raise Exception("Zero weight may cause singularity")
            logging.info('  similar: ' + v + ', w: ' + str(w))
            if v not in g:
                g[v] = {}
                if context:
                    left.add(v)
                else:
                    right.add(v)
            g[u][v] = w
            g[v][u] = w
            if depth < max_depth:
                h.append((v,depth + 1))

    return g, left, right

def effective_conductance_from_graph(g, keyword_phrase, targets):
    d = len(g.keys())
    a = np.zeros((d,d))
    b = np.zeros(d)
    idx = {}
    for i, u in enumerate(g):
        idx[u] = i
    for i, u in enumerate(g):
        edges = g[u]
        for v in edges:
            w = edges[v]
            a[i][idx[v]] = w
            a[idx[v]][i] = w
    c_a = np.sum(a,axis=0)
    oa = a
    a = np.diag(1/c_a).dot(a)
    for j in range(d):
        a[j,j] = -1
    for j in range(d):
        a[idx[keyword_phrase],j] = 0
    a[idx[keyword_phrase],idx[keyword_phrase]] = 1.
    b[idx[keyword_phrase]] = 1.

    res = []
    for target in tqdm.tqdm(targets):
        original_row = np.copy(a[idx[target],:])
        proxy_row = np.zeros(d)
        proxy_row[idx[target]] = 1.
        a[idx[target],:] = proxy_row
        voltages = np.linalg.solve(a,b)
        edges = g[keyword_phrase]
        effective_capacity = 0
        for e in edges:
            effective_capacity += (1 - voltages[idx[e]]) * edges[e]
        res.append((effective_capacity, voltages))
        a[idx[target],:] = original_row
    return res

def effective_conductance_from_corpus(matcher, seed_keyword_phrases, max_depth):
    g, left, right = mk_graph(seed_keyword_phrases, matcher, max_depth)
    # if context_phrase not in list(left):
    #      raise Exception("Context phrase not found during search.")
    targets = set(left)
    # TODO change below to handle multiple seed keyphrases
    results = []
    for seed_keyword_phrase in seed_keyword_phrases:
        results.extend(list(zip(targets, map(lambda x: x[0], effective_conductance_from_graph(g, seed_keyword_phrase, targets)), len(targets) * [seed_keyword_phrase])))
    results.sort(key=lambda x: x[1])
    return results, g, left, right

def print_graph(g,l,r):
    for u in g:
         if u in l:
              type = "context"
         else:
              type = "keyword"
         print("(" + type + ") " + u)
         for v in g[u]:
               if u in l and v in l:
                   color = 'red'
               elif u in r and v in r:
                   color = 'green'
               else:
                   color = 'black'
               print("  (" + color + " " + ('%.2f' % (g[u][v],)) + ") " + v)

def currents(g, voltages):
    res = {}
    for u in g:
        sum = 0
        for v in g[u]:
            sum += (voltages[u] - voltages[v]) * g[u][v]
        res[u] = sum
    return res

def print_graph_info(g,l,r):
    print("{} node(s), {} keyword(s), {} context(s)".format(len(g), len(r), len(l)))
    green = 0
    red = 0
    black = 0
    for u in g:
        for v in g[u]:
            if u <= v:
                if '__' in u and '__' in v:
                    red += 1
                elif '__' not in u and '__' not in v:
                    green += 1
                else:
                    black += 1
    print("{} green edge(s), {} red edge(s), {} black edges".format(green, red, black))

def generated_sentences(results, keyword_phrase, matcher):
    return list(map(lambda x: matcher.combine(x[0], keyword_phrase), results))

def open_editor(initial_message):
    # https://stackoverflow.com/questions/6309587/how-to-launch-an-editor-e-g-vim-from-a-python-script
    EDITOR = os.environ.get('EDITOR','vim')
    with tempfile.NamedTemporaryFile(suffix=".tmp") as tf:
        tf.write(initial_message.encode('utf8'))
        tf.flush()
        call([EDITOR, tf.name])
        tf.seek(0)
        edited_message = tf.read().decode()
    return edited_message

def parse_user_corpus(lines):
    keyphrases = []
    corpus = []
    for line in lines.split('\n'):
        match = re.compile('_(.*)_').search(line)
        if match:
            keyphrase = line[match.span()[0]+1:match.span()[1]-1]
            # context = line[:match.span()[0]+1] + lines[match.span()[1]-1:]
            keyphrases.append(keyphrase)
            corpus.append(line.replace('_',''))
    return keyphrases, corpus

def snowball(corpus, keyphrases, depth):
    matcher = RegexMatch(corpus)
    results, g, l, r = effective_conductance_from_corpus(matcher, keyphrases, depth)
    print_graph_info(g,l,r)
    print()
    print_graph(g,l,r)
    print()
    return results

if __name__ == "__main__":
    # TODO improve performance
    args = parser.parse_args()
    from regex_match import RegexMatch
    initial_message = ''
    all_outputs = []
    if args.debug_seed_files:
        debug_seed_files = args.debug_seed_files.split(" ")
    else:
        debug_seed_files = []
    corpus = load_corpus(args.filename)
    shown_so_far = set([])
    i = 0
    while True:
        if len(debug_seed_files) > i:
            user_corpus = ''.join(open(debug_seed_files[i], 'r').readlines())
        else:
            user_corpus = open_editor(initial_message)
        # add to seen so that user does not see their own output twice
        # do this before parse_user_corpus to keep underscores
        for u in user_corpus.split('\n'):
            # breakpoint()
            shown_so_far.add(u)
        keyphrases, user_corpus = parse_user_corpus(user_corpus)
        all_outputs.extend(user_corpus)
        if args.output:
          with open(args.output, 'w') as f:
              for line in all_outputs:
                  f.write(line + '\n')
        corpus.extend(user_corpus)
        results = snowball(corpus, keyphrases, args.depth)
        # filter so that user does not see same snowball result twice
        candidates = reversed([ a.replace('__', '_' + c + '_') for a,b,c, in results])
        initial_message = []
        for c in candidates:
            # breakpoint()
            if c not in shown_so_far:
                shown_so_far.add(c)
                initial_message.append(c)
        if not initial_message:
            print("No new candidates to show!")
            break
        initial_message = '\n'.join(initial_message)
        for r,w,_ in reversed(results):
            print(w, r)
        i += 1
