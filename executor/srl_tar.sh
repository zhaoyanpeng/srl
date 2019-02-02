flag="keep"
if [ ! -z $1 ]; then
    flag=$1
fi

proot="/afs/inf.ed.ac.uk/user/s18/s1847450/Code/allennlp_shit"
model_name="srl_verb"
param_name="srl_elmo_replicate.jsonnet"
param_path="$proot/mimiconf"
model_path="/disk/scratch1/s1847450/model"

library="nlpmimic"
this_model=$model_path/$model_name

python -m nlpmimic.run archive $param_path/$param_name -s $this_model -r --include-package $library

