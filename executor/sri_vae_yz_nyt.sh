flag="keep"
if [ ! -z $1 ]; then
    flag=$1
fi

proot="/afs/inf.ed.ac.uk/user/s18/s1847450/Code/nlpmimic"
droot="/disk/scratch1/s1847450"
param_path="$proot/mimiconf"

model_name="sri_vae_yz_nyt_c0.1_128_kl.0_re.0_ky.1_ll.1_b.z_sp1.5_rw_n20.0_lemma.ctx"
param_name="sri_vae_yz_nyt.jsonnet"
model_path=$droot/model

library="nlpmimic"
this_model=$model_path/$model_name
log_file=$droot/log/"$model_name".log

echo "Be careful this may delete "$this_model

if [ $flag = "remove" ]; then
    echo "Deleting "$this_model
    rm $this_model/* -rf
    
    nohup python -m allennlp.run train $param_path/$param_name -s $this_model --include-package $library > $log_file 2>&1 & 
elif [ $flag = "recover" ]; then
    nohup python -m allennlp.run train $param_path/$param_name -s $this_model -r --include-package $library >> $log_file 2>&1 &
elif [ $flag = "tune" ]; then
    echo "Deleting "$this_model
    rm $this_model/* -rf
    allennlp train $param_path/$param_name -s $this_model --include-package $library  
else
    allennlp train $param_path/$param_name -s $this_model --include-package $library
fi

