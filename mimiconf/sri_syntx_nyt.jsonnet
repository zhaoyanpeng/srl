{
    "dataset_reader":{
        "type":"conll2009",
        "lazy": true,
        "maximum_length": 80,
        "valid_srl_labels": ["A0", "A1", "A2", "A3", "A4", "A5", "AM-ADV", "AM-CAU", "AM-DIR", 
                             "AM-EXT", "AM-LOC", "AM-MNR", "AM-NEG", "AM-PRD", "AM-TMP"],
        "lemma_file":  "/disk/scratch1/s1847450/data/conll09/all.moved.arg.vocab",
        "lemma_use_firstk": 5,
        "feature_labels": ["pos", "dep"],
        "move_preposition_head": true,
        "max_num_argument": 7,
        "instance_type": "srl_graph",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },

    "nytimes_reader": {
        "type":"conllx_unlabeled",
        "lazy": true,
        "maximum_length": 80,
        "valid_srl_labels": ["A0", "A1", "A2", "A3", "A4", "A5", "AM-ADV", "AM-CAU", "AM-DIR", 
                             "AM-EXT", "AM-LOC", "AM-MNR", "AM-NEG", "AM-PRD", "AM-TMP"],
        "lemma_file":  "/disk/scratch1/s1847450/data/conll09/all.moved.arg.vocab",
        "lemma_use_firstk": 5,
        "feature_labels": ["pos", "dep"],
        "move_preposition_head": false,
        "max_num_argument": 7,
        "min_valid_lemmas": 0.5,
        "instance_type": "srl_graph",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },

    "reader_mode": "srl_nyt",
    "validation_ontraining_data": false,

    "add_unlabeled_noun": true,
    "train_dx_firstk": 500,

    "train_dx_path": "/disk/scratch1/s1847450/data/conll09/morph.word/v20.0/train.verb",
    //"test_data_path": "/disk/scratch1/s1847450/data/conll09/morph.word/v20.0/test.verb",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/morph.word/v20.0/devel.verb",

    //"train_dx_context_path":  "/disk/scratch1/s1847450/data/nytimes/morph.word/v100.0/nytimes.verb.ctx",
    //"train_dx_appendix_path": "/disk/scratch1/s1847450/data/nytimes/morph.word/v100.0/nytimes.verb.sel",

    //"train_dx_path": "/disk/scratch1/s1847450/data/conll09/separated/train.verb",
    "test_data_path": "/disk/scratch1/s1847450/data/conll09/separated/test.verb",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.verb",

    "train_dx_context_path":  "/disk/scratch1/s1847450/data/nytimes/morph.word/v100.0/nytimes.verb.ctx",
    "train_dx_appendix_path": "/disk/scratch1/s1847450/data/nytimes/morph.word/v100.0/nytimes.verb.sel",
    //"train_dx_context_path":  "/disk/scratch1/s1847450/data/nytimes/morph.only/nytimes.45.only",
    //"train_dx_appendix_path": "/disk/scratch1/s1847450/data/nytimes/morph.only/nytimes.45.verb.only",

    "vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src",
    "datasets_for_vocab_creation": ["vocab"],

    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 5 
    },

    "trainer": {
        "type": "sri_synyt",
    }
}
