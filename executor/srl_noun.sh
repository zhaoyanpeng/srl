flag="keep"
if [ ! -z $1 ]; then
    flag=$1
fi

proot="/afs/inf.ed.ac.uk/user/s18/s1847450/Code/allennlp_shit"
droot="/disk/scratch1/s1847450"
param_path="$proot/mimiconf"

model_name="srl_noun_100d_sbilstm_elmo"
param_name="srl_elmo.jsonnet"
model_path=$droot/model

library="nlpmimic"
this_model=$model_path/$model_name
log_file=$droot/log/"$model_name".log

echo "Be careful this may delete "$this_model

if [ $flag = "remove" ]; then
    echo "Deleting "$this_model
    rm $this_model/* -rf

    nohup allennlp train $param_path/$param_name -s $this_model --include-package $library > $log_file 2>&1 &
elif [ $flag = "recover" ]; then
    nohup allennlp train $param_path/$param_name -s $this_model -r --include-package $library >> $log_file 2>&1 &
else
    allennlp train $param_path/$param_name -s $this_model --include-package $library
fi

