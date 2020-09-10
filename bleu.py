from nltk.translate.bleu_score import sentence_bleu
import tqdm
import argparse
import numpy as np
import re

parser = argparse.ArgumentParser()
parser.add_argument("reference")
parser.add_argument("generated")

def load_corpus(filename):
     res = []
     cleanup = re.compile('[^a-zA-Z\s_]')
     for line in open(filename, 'r'):
          line = cleanup.sub(' ', line)
          line = ' '.join(line.lower().split())
          res.append(line)
     return res

def bleu(held_out_sentences, generated_sentences):
    i_max = len(generated_sentences)
    j_max = len(held_out_sentences)
    scores = np.zeros((i_max,j_max))
    best = []
    for i in tqdm.tqdm(range(i_max)):
        for j in range(j_max):
            scores[i][j] = sentence_bleu([held_out_sentences[j].split()], generated_sentences[i].split())
        idx = np.where(scores[i] == np.max(scores[i]))[0][0]
        best.append((generated_sentences[i], held_out_sentences[idx], scores[i][idx]))
    best.sort(key=lambda x: x[2])
    for b in best:
         print(round(b[2],3),b[0],',',b[1])
    return scores.max(axis=1).mean()

if __name__ == "__main__":
    args = parser.parse_args()
    print(round(bleu(load_corpus(args.reference), load_corpus(args.generated)),3))
