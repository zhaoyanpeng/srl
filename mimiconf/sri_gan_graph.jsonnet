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
  
    "reader_mode": "srl_gan",
    "dis_param_name": ["discriminator"],
    "validation_ontraining_data": false,

    "train_dx_path": "/disk/scratch1/s1847450/data/conll09/morph.word/5.0/train.noun",
    "train_dy_path": "/disk/scratch1/s1847450/data/conll09/morph.word/5.0/train.verb",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/morph.word/5.0/devel.noun",

    "model": {
        "discriminator": {
            "type": "sri_gan_dis",
            "encoder": {
                "type": "srl_graph_encoder",
                "input_dim": 100, 
                "layer_timesteps": [2, 2, 2, 2],
                "residual_connection_layers": {"2": [0], "3": [0, 1]},
                "dense_layer_dims": [100],
                "node_msg_dropout": 0.3,
                "residual_dropout": 0.3,
                "aggregation_type": "a",
                "combined_vectors": false,
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
            "feature_matching": true,
            "label_smooth_val": 1.2,
        },

        "generator": {
            "type": "sri_gan_gen",
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

        "type": "sri_gan",
        "straight_through": true,
        "use_uniqueness_prior": false,
        "uniqueness_loss_type": 'unscale_kl"',
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 150 
    },
    "trainer": {
        "type": "sri_gan",
        "num_epochs": 1000,
        "grad_clipping": 1.0,
        "patience": 200,
        "shuffle": true,
        "num_serialized_models_to_keep": 2,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": 1,
        "dis_min_loss": 0.0,
        "dis_skip_nepoch": 0,
        "gen_skip_nepoch": 0,
        "gen_pretraining": -1,  
        "dis_loss_scalar": 0.05,
        "gen_loss_scalar": 1.0,
        "kld_loss_scalar": 0.5,
        "bpr_loss_scalar": 1.0,
        "sort_by_length": true,
        "consecutive_update": false,
        "dis_max_nbatch": 1,
        "gen_max_nbatch": 1,
        "optimizer": {
          "type": "adadelta",
          "rho": 0.95
        },
        "optimizer_dis": {
          "type": "adadelta",
          "rho": 0.95
        }
    }
}
