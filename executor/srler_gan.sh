fold=srl_verb_lyu
model_path=/disk/scratch1/s1847450/model/$fold/model.tar.gz
#context_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.45.lemma.small
#appendix_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.arg.verb.small
#output_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.tag.verb.small
context_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.45.lemma
appendix_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.pre.verb
output_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.tag.verb
bsize=500
cuda_id=0
predictor=srl_naive

library="nlpmimic"

python -m nlpmimic.run srler_gan --batch-size $bsize --cuda-device $cuda_id \
    --output-file $output_file $model_path $context_file $appendix_file --predictor $predictor \
    --include-package $library
