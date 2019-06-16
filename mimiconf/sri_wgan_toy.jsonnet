{
    "vocabulary": {
        "tokens_to_add": {"lemmas": ["NULL_LEMMA"]}
    },
    "dataset_reader":{
        "type":"conll2009",
        "feature_labels": ["pos", "dep"],
        "move_preposition_head": true,
        "max_num_argument": 7,
        "instance_type": "srl_graph"
    },
  
    "reader_mode": "srl_gan",
    "dis_param_name": ["discriminator"],

    "train_dx_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",
    "train_dy_path": "/disk/scratch1/s1847450/data/conll09/bitgan/verb.bit",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",

    "model": {
        "discriminator": {
            "type": "sri_wgan_dis",
            "grad_penal_weight": 1,
            "encoder": {
                "type": "srl_simple_encoder",
                "input_dim": 2, 
                "layer_timesteps": [2, 2, 2, 2],
                //"residual_connection_layers": {"2": [0], "3": [0, 1]},
                "dense_layer_dims": [2],
                "node_msg_dropout": 0.3,
                "residual_dropout": 0.3,
                "aggregation_type": "a",
                "combined_vectors": false,
            },
            "lemma_embedder": {
                "token_embedders": {
                    "lemmas": {
                        "type": "embedding",
                        "embedding_dim": 2,
                        "vocab_namespace": "lemmas",
                        "trainable": false 
                    }
                }
            },
            "label_embedder": {
                "embedding_dim": 2,
                "vocab_namespace": "srl_tags",
                "trainable": true,
                "sparse": false 
            },
            "predt_embedder": {
                "embedding_dim": 2,
                "vocab_namespace": "predicates",
                "trainable": true, 
                "sparse": false 
            },
        },

        "generator": {
            "type": "sri_gan_gen",
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
            "seq_encoder": {
                "type": "stacked_bidirectional_lstm",
                "input_size": 4,
                "hidden_size": 2,
                "num_layers": 1,
                "recurrent_dropout_probability": 0.0,
                "use_highway": true
            },
            "tau": 0.01,
            "tunable_tau": false,
            "psign_dim": 2,
            "seq_projection_dim": null,
            "embedding_dropout": 0.0,
            "suppress_nonarg": true,
        },

        "type": "sri_gan",
        "straight_through": true,
        "use_uniqueness_prior": true,
        "uniqueness_loss_type": 'unscale_kl"',
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 3 
    },
    "trainer": {
        "type": "sri_gan",
        "num_epochs": 5,
        "grad_clipping": 1.0,
        "patience": 20,
        "shuffle": false,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": 3,
        "dis_min_loss": -100000.45,
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
        },
        // wgan
        "clip_val": 0.01,
        "optimizer_wgan": {
          "type": "adam",
          "lr": 1e-3
        },
        "optimizer_wgan_dis": {
          "type": "rmsprop",
          "lr": 1e-3
        },
    }
}
