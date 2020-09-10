# Mushroom

*Mushroom* helps data scientists synthesize utterances for training text classifiers. In the intended use case, the data scientist wishes to synthesize utterances from some target class and has at their disposal an existing corpus of utterances which do not belong to the target class. The data scientist provides the corpus as well as a small number of seed utterances from the target label class, and *Mushroom* does the rest, leveraging the corpus to output a synthesized set of utterances.

# Scripts

## mushroom.py

### Arguments

#### filename

#### context_phrase

#### keyword_phrase

### Optional arguments

#### depth

#### output

### Example invocations

python mushroom.py atis.txt "what is the __" "luggage limit" --output results.txt

## bleu.py

### Arguments

#### reference_filename

#### generated_filename

#### 

## baseline.py

### Arguments

#### filename

#### keyword_phrase

#### n
