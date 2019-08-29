fold=srl_noun_moved
model_path=/disk/scratch1/s1847450/model/$fold/model.tar.gz

context_file=/disk/scratch1/s1847450/data/nytimes.new/morph.only/nytimes.45.only
appendix_file=/disk/scratch1/s1847450/data/nytimes.new/morph.only/nytimes.45.noun.pred
output_file=/disk/scratch1/s1847450/data/nytimes.new/morph.only/nytimes.45.noun.only
indxes_file=/disk/scratch1/s1847450/data/nytimes.new/morph.only/nytimes.45.noun.pred.cnt
log_file=/disk/scratch1/s1847450/data/nytimes.new/morph.only/nytimes.45.noun.only.it.log

#context_file=/disk/scratch1/s1847450/data/nytimes.new/morph.only/nytimes.45.only
#appendix_file=/disk/scratch1/s1847450/data/nytimes.new/morph.only/nytimes.45.verb.pred
#output_file=/disk/scratch1/s1847450/data/nytimes.new/morph.only/nytimes.45.verb.only
#indxes_file=/disk/scratch1/s1847450/data/nytimes.new/morph.only/nytimes.45.verb.pred.cnt
#log_file=/disk/scratch1/s1847450/data/nytimes.new/morph.only/nytimes.45.verb.only.it.log

#context_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.45.lemma.small
#appendix_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.pre.verb.small
#output_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.tag.verb.small.pred
#indxes_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.pre.verb.cnt.small
#log_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.tag.verb.small.log

#context_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.45.lemma
#appendix_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.pre.verb
#output_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.tag.verb.pred
#indxes_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.pre.verb.cnt
#log_file=/disk/scratch1/s1847450/data/nyt_annotated/xchen/nytimes.tag.verb.log

bsize=300
cuda_id=2
predictor=srl_naive

library="nlpmimic"

nohup python -m nlpmimic.run srler_nyt --batch-size $bsize --cuda-device $cuda_id \
    --output-file $output_file --indxes-file $indxes_file $model_path $context_file $appendix_file \
    --predictor $predictor --include-package $library > $log_file 2>&1 &
