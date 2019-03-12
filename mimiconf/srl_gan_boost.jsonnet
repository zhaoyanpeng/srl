{
    "dataset_reader": {
        "type": "conll2009",
        "feature_labels": [
            "pos",
            "dep"
        ],
        "instance_type": "srl_gan",
        "move_preposition_head": true,
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 64,
        "padding_noise": 0,
        "sorting_keys": [
            [
                "tokens",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "srl_gan",
        "binary_feature_dim": 100,
        "fixed_temperature": false,
        "initializer": [
            [
                "tag_projection_layer.*weight",
                {
                    "type": "orthogonal"
                }
            ]
        ],
        "label_embedder": {
            "embedding_dim": 100,
            "sparse": false,
            "trainable": true,
            "vocab_namespace": "srl_tags"
        },
        "lemma_embedder": {
            "token_embedders": {
                "lemmas": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "pretrained_file": "/disk/scratch1/s1847450/data/lemmata/en.lemma.300.20.vec.c2009",
                    "trainable": false,
                    "vocab_namespace": "lemmas"
                }
            }
        },
        "mask_empty_labels": false,
        "predicate_embedder": {
            "embedding_dim": 100,
            "sparse": false,
            "trainable": true,
            "vocab_namespace": "predicates"
        },
        "seq_encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 600,
            "input_size": 1124,
            "num_layers": 3,
            "recurrent_dropout_probability": 0.3,
            "use_highway": true
        },
        "srl_encoder": {
            "type": "srl_gan_dis",
            "attent_size": 200,
            "embedding_dim": 500,
            "hidden_size": 200,
            "module_choice": "c",
            "num_layer": 1,
            "projected_dim": 200
        },
        "temperature": 1,
        "token_embedder": {
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "do_layer_norm": false,
                    "dropout": 0,
                    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
                }
            }
        },
        "zero_null_lemma_embedding": true
    },
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.noun",
    "trainer": {
        "type": "srl_gan",
        "cuda_device": 2,
        "dis_loss_scalar": 0.05,
        "dis_skip_nepoch": 500,
        "gen_loss_scalar": 1,
        "gen_pretraining": 0,
        "gen_skip_nepoch": 0,
        "grad_clipping": 1,
        "num_epochs": 500,
        "num_serialized_models_to_keep": 10,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        },
        "optimizer_dis": {
            "type": "adadelta",
            "rho": 0.95
        },
        "patience": 50,
        "shuffle": true,
        "validation_metric": "+f1-measure-overall"
    },
    "vocabulary": {
        "tokens_to_add": {
            "lemmas": [
                "NULL_LEMMA"
            ]
        }
    },
    "datasets_for_vocab_creation": [
        "vocab"
    ],
    "dis_param_name": ["srl_encoder"],
    "reader_mode": "srl_gan",
    "train_dx_path": "/disk/scratch1/s1847450/data/conll09/separated/noun.morph.picked",
    "train_dy_path": "/disk/scratch1/s1847450/data/conll09/separated/verb.morph.picked",
    "vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src"
}
