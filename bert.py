from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
import tqdm
import math
import numpy as np

model_name = 'bert-large-uncased'
bert_tokeniser = BertTokenizer.from_pretrained(model_name)
bert_model = BertForMaskedLM.from_pretrained(model_name)

def do_all(example,number_to_remove,number_to_insert):
    results = []
    tokens = bert_tokeniser.tokenize(f"[CLS] {example} [SEP]")
    for i in range(1,len(tokens)-1):
        for j in range(number_to_remove):
            for k in range(number_to_insert):
                lo = i-j
                if lo <= 0:
                    continue
                mask_idxs = list(range(lo,lo+k+1))
                tokens_with_mask = tokens[:lo] + (len(mask_idxs) * ["[MASK]"]) + tokens[i:]
                token_idxs = torch.tensor([bert_tokeniser.convert_tokens_to_ids(tokens_with_mask)])
                segment_ids = torch.tensor([[0]*len(token_idxs)])
                with torch.no_grad():
                    output = bert_model(token_idxs, segment_ids)
                output_tokens = np.array(bert_tokeniser.convert_ids_to_tokens(token_idxs.tolist()[0]))
                output_tokens[mask_idxs] = bert_tokeniser.convert_ids_to_tokens(torch.argmax(output[0, mask_idxs],axis=1).tolist())
                # print(tokens_with_mask, output_tokens)
                results.append(' '.join(output_tokens[1:-1]).replace("##",""))
    results.append(example)
    return results
    # TODO rank results

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn

from string import punctuation

def _clean_word(word): return word.lower().strip(punctuation)

def _convert_nltk_to_wordnet_tag(pos_tag):
  if pos_tag.startswith("N"):
    return wn.NOUN
  if pos_tag.startswith("V"):
    return wn.VERB
  if pos_tag.startswith("R"):
    return wn.ADV
  if pos_tag.startswith("J"):
    return wn.ADJ

def synonyms(word, pos_tag):
  return list({
      lemma.replace("_"," ").replace("-"," ") for synset in wn.synsets(
        _clean_word(word),
        pos_tag,
      ) for lemma in synset.lemma_names()
    })

def _infer_pos_tags(example):
    results = []
    tokens = nltk.tokenize.casual_tokenize(example)
    for i, (token, nltk_tag) in enumerate(nltk.pos_tag(tokens)):
        tag = _convert_nltk_to_wordnet_tag(nltk_tag)
        if tag is not None:
            for s in synonyms(token, tag):
                r = list(tokens)
                r[i] = s
                results.append(' '.join(r))
    return results

def syn_then_do(example):
    res = list(set([j for i in tqdm.tqdm(_infer_pos_tags(example)) for j in do_all(i,2,2) ]))
    res.sort(key=get_score)
    return res

# Or better yet a workshop paper that turns this into a beam search with primitives for example you have primitive operations like insert a span, remove a span, find a synonym and then you have; and then you have a beam that is scored based on likelihood (fluency) and similarity (faithfulness) which are standard MT goals.
# the nice thing here is that these actions are complimentary, you can add a synonym 'find me a ticket to nyc' -> 'obtain me a ticket to nyc' -> 'obtain for me a ticket to nyc'
# note also you don't have to take the argmax from bert but can sample from it
# another primitive would be back-translating from an embedding
# if you can't balance the beam score, you can perhaps alternate faithfulness
# and maybe you can keep the "best of all time"
# would be good to do an autoencoder type of embed and decode primitive

# def get_score(sentence):
#     tokenize_input = bert_tokeniser.tokenize(sentence)
#     tensor_input = torch.tensor([bert_tokeniser.convert_tokens_to_ids(tokenize_input)])
#     predictions=bert_model(tensor_input)
#     loss_fct = torch.nn.CrossEntropyLoss()
#     loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data 
#     return math.exp(loss)

from gpt2_client import GPT2Client
gpt2 = GPT2Client('774M')
gpt2.load_model(force_download=False)

from transformers import GPT2Tokenizer, GPT2LMHeadModel
with torch.no_grad():
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model.eval()
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def get_score(sentence):
    tokenize_input = gpt2_tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input])
    loss=gpt2_model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())

# syn_then_do("please cancel my restaurant reservations")

def beam_search_decoder(data, k):
	sequences = [[list(), 0.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score - log(row[j])]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences

from sentence_transformers import SentenceTransforme
sim_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def similarity(a,b):
    res = sim_model.encode([a,b])
    return 1 / (1 + np.linalg.norm(res[0] - res[1]))
