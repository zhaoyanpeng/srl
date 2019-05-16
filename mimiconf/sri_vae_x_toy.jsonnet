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

    "train_dx_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",
    "train_dy_path": "/disk/scratch1/s1847450/data/conll09/bitgan/verb.bit",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",

    "model": {
        "autoencoder": {
            "type": "srl_lstms_ae",
            "decoder": {
                "type": "srl_lstms_decoder",
                "input_dim": 4, //  2 + 2
                "hidden_dim": 3, 
            },
        },

        "classifier": {
            "type": "srl_vae_classifier",
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

        "type": "srl_vae_d",
        "alpha": 0.5,
        "nsampling": 2,
        "straight_through": true,

    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 5 
    },
    "trainer": {
        "type": "sri_vae",
        "num_epochs": 2,
        "grad_clipping": 1.0,
        "patience": 20,
        "shuffle": false,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": -1,
        "gen_loss_scalar": 1.0,
        "kld_loss_scalar": 0.5,
        "kld_update_rate": 0.05,
        "kld_update_unit": 5,
        "bpr_loss_scalar": 1.0,
        "sort_by_length": false,
        "shuffle_arguments": true,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        },
    }
}
