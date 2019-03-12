flag="keep"
if [ ! -z $1 ]; then
    flag=$1
fi

proot="/afs/inf.ed.ac.uk/user/s18/s1847450/Code/nlpmimic"
droot="/disk/scratch1/s1847450"
param_path="$proot/mimiconf"

model_name="srl_gan_boost"
param_name="srl_gan_boost.jsonnet"
model_path=$droot/model

library="nlpmimic"
this_model=$model_path/$model_name
log_file=$droot/log/"$model_name".log

echo "Be careful this may delete "$this_model

if [ $flag = "remove" ]; then
    echo "Only recover option supported"
    #echo "Deleting "$this_model
    #rm $this_model/* -rf

    #nohup python -m nlpmimic.run boost $param_path/$param_name -s $this_model --include-package $library > $log_file 2>&1 &
elif [ $flag = "recover" ]; then
    nohup python -m nlpmimic.run boost $param_path/$param_name -s $this_model -r --include-package $library >> $log_file 2>&1 &
elif [ $flag = "tune" ]; then
    echo "Only recover option supported"
else
    echo "Only recover option supported"
fi

