rm -rf benchmarks/results
mkdir -p benchmarks/results

set -e
set -o xtrace

trim() {
    echo "${1}" | gawk '{gsub(/^[ \t]+/,""); print $0}'
}

while IFS=, read -r corpus label context keyphrase
do
  label=$(trim "${label}")
  context=$(trim "${context}")
  keyphrase=$(trim "${keyphrase}")
  whole=${context/__/$keyphrase}
  holdouts=benchmarks/results/"${label}".holdouts.txt
  debug=benchmarks/results/"${label}".debug.txt
  synthesized=benchmarks/results/"${label}".synthesized.txt
  train=benchmarks/results/"${label}".train.txt
  grep -v "^$label" $corpus | csvcut -S -c 2 > "$train"
  python mushroom.py "$train" "$context" "$keyphrase"  --output "$synthesized" > "$debug"
  grep "^$label" $corpus | csvcut -S -c 2 | grep -v "$whole" > "$holdouts"
  python bleu.py "$holdouts" "$synthesized" > benchmarks/results/"${label}".evaluation.txt
  cat benchmarks/results/"${label}".evaluation.txt | tail -n 1 | awk -v q=", $label" '{print $0 q}' >> benchmarks/results/scores.txt
done < benchmarks/benchmark.csv

awk '{ total += $1; count++ } END { print total/count }' benchmarks/results/scores.txt >> benchmarks/results/scores.txt
