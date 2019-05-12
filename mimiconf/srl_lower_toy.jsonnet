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
    },
  
    "reader_mode": "basic",
    "validation_ontraining_data": false,

    "train_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",
    "test_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/verb.bit",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",

    "model": {
        "type": "srl_lower",
        "token_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 2,
                    "vocab_namespace": "tokens",
                    "trainable": true 
                }
            }
        },
        "outputfile": "/disk/scratch1/s1847450/vectors",
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 3 
    },
    "trainer": {
        "type": "srl_lower",
        "num_epochs": 1,
        "grad_clipping": 1.0,
        "patience": 1,
        "shuffle": false,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": -1,
        "gen_loss_scalar": 1.0,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        },
    }
}
