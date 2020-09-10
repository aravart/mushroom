import argparse
import numpy as np
import re

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("keyword_phrase")
parser.add_argument("n", type=int)

def load_corpus(filename):
     res = []
     cleanup = re.compile('[^a-zA-Z\s_]')
     for line in open(filename, 'r'):
          line = cleanup.sub(' ', line)
          line = ' '.join(line.lower().split())
          res.append(line)
     return res

def baseline(corpus, keyword_phrase, n):
    target_length = len(keyword_phrase.split())
    res = []
    while len(res) < n:
         idx = np.random.choice(len(corpus))
         candidate = corpus[idx]
         if len(candidate) > target_length:
              candidate = candidate.split()
              bad_seed_context = ' '.join(candidate[:-target_length])
              res.append(bad_seed_context + ' ' + keyword_phrase)
    return res

if __name__ == "__main__":
    args = parser.parse_args()
    corpus = load_corpus(args.filename)
    generated = baseline(corpus, args.keyword_phrase, args.n)
    for sentence in generated:
        print(sentence)
