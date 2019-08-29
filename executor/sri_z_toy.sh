root="/afs/inf.ed.ac.uk/user/s18/s1847450/Code/nlpmimic"
flag="keep"
if [ ! -z $1 ]; then
    flag=$1
fi

model_name="sri_z_toy"
param_name="sri_z_toy.jsonnet"
param_path="$root/mimiconf"
model_path="/disk/scratch1/s1847450/model"

library="nlpmimic"
this_model=$model_path/$model_name

echo "Be careful this may delete "$this_model

if [ $flag = "remove" ]; then
    echo "Deleting "$this_model
    rm $this_model/* -rf
    allennlp train $param_path/$param_name -s $this_model --include-package $library 
else
    allennlp train $param_path/$param_name -s $this_model -r --include-package $library 
fi

