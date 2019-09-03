#fold=sri_vae_r_hard_alpha5.5_128_sp2.10_-c_rw_ap_p_l5_v20.0
#name=sri_vae_r_5.5_l5
#word=v20.0
#data=devel
#post=verb
fold=sri_vae_y_+nyt_c.5_128_sp0.10_rw_n20.0_le.20_lemma.ctx
name=nyt
word=sri
data=train.noun.morph
post=only
#post=noun.morph.picked
model_path=/disk/scratch1/s1847450/model/$fold/model.tar.gz
#input_file=/disk/scratch1/s1847450/data/conll09/morph.word/"$word"/"$data"."$post"
input_file=/disk/scratch1/s1847450/data/conll09/morph.only/"$data"."$post"
output_file=/disk/scratch1/s1847450/data/conll09/"$word"_"$name"_"$data"_"$post"
bsize=150
cuda_id=1
predictor=srl_naive

library="nlpmimic"

# https://stackoverflow.com/a/43373520
reader=$(cat <<EOF
{"dataset_reader": {
    "type": "conll2009",
    "feature_labels": [
        "pos",
        "dep"
    ],
    "flatten_number": true,
    "instance_type": "srl_graph",
    "lemma_file": "/disk/scratch1/s1847450/data/conll09/morph.only/all.morph.only.moved.arg.vocab.json",
    "lemma_use_firstk": 20,
    "max_num_argument": 7,
    "moved_preposition_head": [
        "IN",
        "TO"
    ],
    "token_indexers": {
        "elmo": {
            "type": "elmo_characters"
        }
    },
    "valid_srl_labels": [
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "AM-ADV",
        "AM-CAU",
        "AM-DIR",
        "AM-EXT",
        "AM-LOC",
        "AM-MNR",
        "AM-NEG",
        "AM-PRD",
        "AM-TMP"
    ]
}}
EOF
)

python -m nlpmimic.run srler --batch-size $bsize --cuda-device $cuda_id \
    --output-file $output_file $model_path $input_file --predictor $predictor \
    --include-package $library -o "$reader"
