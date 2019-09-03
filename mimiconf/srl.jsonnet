{
    "vocabulary": {
        "tokens_to_add": {"lemmas": ["NULL_LEMMA"]}
    },
    "train_dataset_reader":{
        "type":"conll2009",
        //"maximum_length": 80,
        //"valid_srl_labels": ["A1", "A0", "A2", "AM-TMP", "A3", "AM-MNR", "AM-LOC", "A4"],
        //"lemma_file":  "/disk/scratch1/s1847450/data/conll09/all.moved.arg.vocab",
        //"lemma_use_firstk": 20,
        "feature_labels": ["pos", "dep"],
        "moved_preposition_head": ["IN", "TO"],
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

    //"train_data_path": "/disk/scratch1/s1847450/data/conll09/morph.word/v20.0/train.verb",
    //"test_data_path": "/disk/scratch1/s1847450/data/conll09/morph.word/v20.0/test.verb",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/morph.word/v20.0/devel.verb",

    "train_data_path": "/disk/scratch1/s1847450/data/conll09/CoNLL2009-ST-English/CoNLL2009-ST-English-train.txt",
    "test_data_path": "/disk/scratch1/s1847450/data/conll09/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt",

    //"train_data_path": "/disk/scratch1/s1847450/data/conll09/separated/train.verb",
    //"test_data_path": "/disk/scratch1/s1847450/data/conll09/separated/test.verb",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.verb",

    //"train_data_path": "/disk/scratch1/s1847450/data/conll09/separated/train.noun",
    //"test_data_path": "/disk/scratch1/s1847450/data/conll09/separated/test.noun",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.noun",

    "vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src",
    "datasets_for_vocab_creation": ["vocab"],

    "model": {
        "type": "srl_model",
        "classifier": {
            "type": "srl_vae_classifier",
            "token_embedder": {
                "token_embedders": {
                    "elmo": {
                        "type": "elmo_token_embedder",
"options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
"weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                        "do_layer_norm": false,
                        "dropout": 0.0
                    }
                }
            },
            "lemma_embedder": {
                "token_embedders": {
                    "lemmas": {
                        "type": "embedding",
                        "embedding_dim": 100,
                        "pretrained_file": "/disk/scratch1/s1847450/data/lemmata/en.lemma.100.20.vec.morph",
                        "vocab_namespace": "lemmas",
                        "trainable": true 
                    }
                }
            },
            "label_embedder": {
                "embedding_dim": 100,
                "vocab_namespace": "srl_tags",
                "trainable": true,
                "sparse": false 
            },
            "predt_embedder": {
                "embedding_dim": 100,
                "vocab_namespace": "predicates",
                "trainable": true, 
                "sparse": false 
            },
            "seq_encoder": {
                "type": "stacked_bidirectional_lstm",
                "input_size": 1124,
                "hidden_size": 600,
                "num_layers": 3,
                "recurrent_dropout_probability": 0.3,
                "use_highway": true
            },
            "initializer": [
              [
                "label_projection_layer.*weight",
                {
                  "type": "orthogonal"
                }
              ]
            ],
            "regularizer": [
                [
                    ".*scalar_parameters.*",
                    {
                        "type": "l2",
                        "alpha": 0.001
                    }
                ]
            ],
            "psign_dim": 100,
            "seq_projection_dim": null,
            "token_dropout": 0.3,
            "lemma_dropout": 0.3,
            "label_dropout": 0.3,
            "predt_dropout": 0.3,
            "suppress_nonarg": false,
        },

    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 128 
    },
    "trainer": {
        "type": "sri_upper",
        "num_epochs": 1000,
        "grad_norm": 5.0,
        //"grad_clipping": 1.0,
        "patience": 200,
        "shuffle": true,
        "num_serialized_models_to_keep": 3,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": 3,
        "gen_loss_scalar": 1.0,
        "optimizer": {
            "type": "dense_sparse_adam",
            "betas": [0.9, 0.9]
            //"type": "adadelta",
            //"rho": 0.95
        },
    }
}
