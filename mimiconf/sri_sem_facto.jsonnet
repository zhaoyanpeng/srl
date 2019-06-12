{
    "vocabulary": {
        "tokens_to_add": {"lemmas": ["NULL_LEMMA"]}
    },
    "dataset_reader":{
        "type":"conll2009",
        "maximum_length": 100,
        //"valid_srl_labels": ["A1", "A0", "A2", "AM-TMP", "A3", "AM-MNR", "AM-LOC", "A4"],
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
  
    "reader_mode": "srl_gan",
    "validation_ontraining_data": false,

    "train_dx_path": "/disk/scratch1/s1847450/data/conll09/morph.word/v20.0/train.noun",
    "train_dy_path": "/disk/scratch1/s1847450/data/conll09/morph.word/v20.0/train.verb",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/morph.word/v20.0/devel.noun",

    //"vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src",
    //"datasets_for_vocab_creation": ["vocab"],

    "model": {
        "autoencoder": {
            "type": "srl_lemma_ae",
            "decoder": {
                "type": "srl_basic_decoder",
                "input_dim": 200, // predicate + label,
                "dense_layer_dims": [450, 600],
                "dropout": 0.1,
            },
        },
        "classifier": {
            "type": "srl_vae_classifier",
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
            "tau": 1.0,
            "tunable_tau": false,
            "psign_dim": 100,
            "seq_projection_dim": null,
            "embedding_dropout": 0.1,
            "suppress_nonarg": true,
        },

        "type": "srl_vae_facto",
        "alpha": 0.5,
        "nsampling": 10,
        "reweight": true, 
        "coupled_loss": true,
        "straight_through": true,
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 128 
    },
    "trainer": {
        "type": "sri_vae",
        "num_epochs": 1000,
        "grad_norm": 5.0,
        "grad_clipping": 1.0,
        "patience": 200,
        "shuffle": true,
        "num_serialized_models_to_keep": 3,
        //"validation_metric": "+f1-measure-overall",
        "validation_metric": "-loss",
        "cuda_device": 1,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        },
    }
}
