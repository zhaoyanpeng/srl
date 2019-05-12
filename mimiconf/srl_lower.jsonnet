{
    "vocabulary": {
        "tokens_to_add": {"lemmas": ["NULL_LEMMA"]}
    },
    "dataset_reader":{
        "type":"conll2009",
        //"maximum_length": 80,
        //"valid_srl_labels": ["A1", "A0", "A2", "AM-TMP", "A3", "AM-MNR", "AM-LOC", "A4"],
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
  
    "reader_mode": "basic",
    "validation_ontraining_data": false,

    //"train_data_path": "/disk/scratch1/s1847450/data/conll09/CoNLL2009-ST-English/CoNLL2009-ST-English-train.txt",
    //"test_data_path": "/disk/scratch1/s1847450/data/conll09/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt",

    //"train_data_path": "/disk/scratch1/s1847450/data/conll09/separated/train.verb",
    //"test_data_path": "/disk/scratch1/s1847450/data/conll09/separated/test.verb",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.verb",

    "train_data_path": "/disk/scratch1/s1847450/data/conll09/separated/train.noun",
    //"test_data_path": "/disk/scratch1/s1847450/data/conll09/separated/test.noun",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.noun",

    "vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src",
    "datasets_for_vocab_creation": ["vocab"],

    "model": {
        "type": "srl_lower",
        "token_embedder": {
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.0
                }
            }
        },
        "outputfile": "/disk/scratch1/s1847450/noun.vec.train",

    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 128 
    },
    "trainer": {
        "type": "srl_lower",
        "num_epochs": 1,
        "grad_clipping": 1.0,
        "patience": 1,
        "shuffle": true,
        "num_serialized_models_to_keep": 1,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": 0,
        "gen_loss_scalar": 1.0,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        },
    }
}
