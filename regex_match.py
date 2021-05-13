import re

def jaccard(u,v):
    u = set(u.split())
    v = set(v.split())
    u.discard('__')
    v.discard('__')
    return len(u.intersection(v)) / len(u.union(v))

class RegexMatch():
     def __init__(self, corpus):
          self.corpus = corpus
          self.counts = {}
          for utterance in corpus:
               if utterance in self.counts:
                    self.counts[utterance] = self.counts[utterance] + 1
               else:
                    self.counts[utterance] = 1

     def context_to_keyphrases(self, context):
          return self.match(context, self.corpus, True)

     def keyphrase_to_contexts(self, keyphrase):
          return self.match(keyphrase, self.corpus, False)

     def keyphrase_to_keyphrases(self, u):
          return self.similar(u, self.corpus)

     def context_to_contexts(self, u):
          return self.similar(u, self.corpus)

     def combine(self, context, keyphrase):
          return context.replace("__", keyphrase)

     def match(self, u, corpus, whole):
          context = '__' in u
          # re.escape(u) might be correct but causes all kinds of performance problems
          # But see python human.py benchmarks/datasets/frames.csv --debug-seed-files "seed1.txt" as there are differences
          try:
            if whole:
                context_pattern = re.compile("\\b" + u.replace('__', '(.*)') + '$')
            else:
                context_pattern = re.compile("\\b" + u.replace('__', '(.*)'))
            keyword_pattern = re.compile("\\b" + u + "\\b")
          except Exception as e:
              print("Fail! " + u)
              return
          for line in corpus:
               if context:
                    if whole:
                         m = context_pattern.match(line)
                    else:
                         m = context_pattern.search(line)
                    if m:
                         yield m.group(1)
               else:
                    if u in line and u != line:
                         m = keyword_pattern.search(line)
                         if m:
                              i = m.span()[0]
                              yield line[:i] + '__' + line[i+len(u):]

     def similar(self, u, corpus):
          context = '__' in u
          if context:
               # Find the last word before __
               s = u.split()
               s = s[s.index('__')-1]
               for v in self.match('__ ' + s, corpus, False):
                    # Now v should *not* contain s so we have to add it back
                    sim = v + ' ' + s + ' __'
                    j = jaccard(sim, u)
                    if sim != u and j != 0: yield sim, j
          else:
               first_word = u.split()[0]
               for v in self.match(first_word + ' __', corpus, False):
                    sim = first_word + ' ' + v
                    j = jaccard(sim, u)
                    if sim != u and j != 0: yield sim, j
