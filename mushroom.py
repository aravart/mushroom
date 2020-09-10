from collections import deque
from nltk.translate.bleu_score import sentence_bleu
import io
import numpy as np
import tqdm
import argparse
import pydot
import logging
import re
import subprocess
import sys
import nltk

stopwords = set(nltk.corpus.stopwords.words('english')) 

if sys.version_info <= (3, 0):
     print("Please use Python 3. This script does not perform correctly on Python 2.")
     sys.exit(0)

epsilon = 0.01

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("context_phrase")
parser.add_argument("keyword_phrase")
parser.add_argument("-d", "--debug", action='store_true')
parser.add_argument("--depth", default=2, type=int)
parser.add_argument("--output", default=None)

logging.basicConfig(level=logging.WARN, format='%(message)s')

def jaccard(u,v):
    u = set(u.split())
    v = set(v.split())
    u.discard('__')
    v.discard('__')
    return len(u.intersection(v)) / len(u.union(v))

def match(u, corpus, whole):
    context = '__' in u
    pattern = re.compile("\\b" + u.replace('__', '(.*)'))
    for line in corpus:
        if context:
            if whole:
                m = pattern.match(line)
            else:
                m = pattern.search(line)
            if m:
                yield m.group(1),line
        else:
            if u in line and u != line:
                i = line.index(u)
                yield line[:i] + '__' + line[i+len(u):],line

def similar(u, corpus):
    context = '__' in u
    if context:
        # Find the last word before __
        s = u.split()
        s = s[s.index('__')-1]
        for v,_ in match('__ ' + s, corpus, False):
            # Now v should *not* contain s so we have to add it back
            sim = v + ' ' + s + ' __'
            j = jaccard(sim, u)
            if sim != u and j != 0: yield sim, j
    else:
        first_word = u.split()[0]
        for v,_ in match(first_word + ' __', corpus, False):
            sim = first_word + ' ' + v
            j = jaccard(sim, u)
            if sim != u and j != 0: yield sim, j

def load_corpus(filename):
     res = []
     cleanup = re.compile('[^a-zA-Z\s_]')
     for line in open(filename, 'r'):
          line = cleanup.sub(' ', line)
          line = ' '.join(line.lower().split())
          res.append(line)
     return res

def mk_graph(keyword_phrase, corpus, max_depth):
    counts = {}
    for utterance in corpus:
        if utterance in counts:
            counts[utterance] = counts[utterance] + 1
        else:
            counts[utterance] = 1

    h = deque([(keyword_phrase, 1)])
    closed = set()
    g = {keyword_phrase: {}}
    left = set()
    right = set([keyword_phrase])

    while h:
        u, depth = h.popleft()
        if u in closed:
            continue
        else:
            closed.add(u)
        context = '__' in u
        if context:
            logging.info('pop (context): ' + u)
        else:
            logging.info('pop (keyword): ' + u)
        for v,_ in match(u, corpus, context):
            logging.info('  match: ' + v)
            if v not in g:
                g[v] = {}
                if context:
                    right.add(v)
                else:
                    left.add(v)
            if context:
                t = u.replace("__", v)
            else:
                t = v.replace("__", u)
            g[u][v] = epsilon * counts[t]
            g[v][u] = epsilon * counts[t]
            if depth < max_depth:
                h.append((v,depth + 1))
        for v,w in similar(u, corpus):
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

def effective_conductance_from_corpus(corpus, keyword_phrase, context_phrase, max_depth):
    g, left, right = mk_graph(keyword_phrase, corpus, max_depth)
    targets = set(list(left) + [context_phrase])
    results = list(zip(targets, map(lambda x: x[0], effective_conductance_from_graph(g, keyword_phrase, targets))))
    results.sort(key=lambda x: x[1])
    return results, g, left, right

def print_graph(g,l,r):
    nodes = {}
    for u in g:
        nodes[u] = pydot.Node(u)
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

def generated_sentences(results, keyword_phrase):
    return list(map(lambda x: x[0].replace("__", keyword_phrase), results))

if __name__ == "__main__":
    args = parser.parse_args()
    corpus = load_corpus(args.filename)
    corpus.append(args.context_phrase.replace("__", args.keyword_phrase))
    results, g, l, r = effective_conductance_from_corpus(corpus, args.keyword_phrase, args.context_phrase, args.depth)
    print_graph_info(g,l,r)
    print()
    print_graph(g,l,r)
    print()
    for r,w in results:
        print(w, r)
    if args.output is not None:
        with open(args.output, 'w') as f:
             for sentence in generated_sentences(results, args.keyword_phrase):
                  f.write(sentence + '\n')
