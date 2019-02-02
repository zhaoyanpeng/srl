fold=srl_verb
model_path=/disk/scratch1/s1847450/model/$fold/model.tar.gz
#input_file=/disk/scratch1/s1847450/data/conll09/separated/devel.verb
#input_file=/disk/scratch1/s1847450/data/conll09/separated/test.verb
input_file=/disk/scratch1/s1847450/data/conll09/hyena.bit
output_file=/disk/scratch1/s1847450/data/conll09/"$fold"_hyena
bsize=150
cuda_id=3
predictor=srl_naive

library="nlpmimic"

python -m nlpmimic.run srler --batch-size $bsize --cuda-device $cuda_id \
    --output-file $output_file $model_path $input_file --predictor $predictor \
    --include-package $library
