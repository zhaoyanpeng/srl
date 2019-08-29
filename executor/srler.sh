#fold=sri_vae_r_hard_alpha5.5_128_sp2.10_-c_rw_ap_p_l5_v20.0
#name=sri_vae_r_5.5_l5
#word=v20.0
#data=devel
#post=verb
fold=srl
name=both
word=srl
data=devel
post=noun
#post=noun.morph.picked
model_path=/disk/scratch1/s1847450/model/$fold/model.tar.gz
#input_file=/disk/scratch1/s1847450/data/conll09/morph.word/"$word"/"$data"."$post"
input_file=/disk/scratch1/s1847450/data/conll09/morph.only/"$data"."$post"
output_file=/disk/scratch1/s1847450/data/conll09/"$word"_"$name"_"$data"_"$post"
bsize=150
cuda_id=3
predictor=srl_naive

library="nlpmimic"

python -m nlpmimic.run srler --batch-size $bsize --cuda-device $cuda_id \
    --output-file $output_file $model_path $input_file --predictor $predictor \
    --include-package $library
