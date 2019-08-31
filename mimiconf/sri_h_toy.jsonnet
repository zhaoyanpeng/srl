{
    "vocabulary": {
        "tokens_to_add": {"tokens": ["NULL_TOKEN"], "lemmas": ["NULL_LEMMA"]}
    },
    "train_dataset_reader":{
        "type":"conll2009",
        "feature_labels": ["pos", "dep"],
        "moved_preposition_head": ["IN", "TO"],
        "max_num_argument": 7,
        "instance_type": "srl_graph"
    },
    "nytimes_reader": {
        "type":"conllx_unlabeled",
        "feature_labels": ["pos", "dep"],
        "max_num_argument": 7,
        "instance_type": "srl_graph",
    },
  
    "reader_mode": "srl_nyt",

    "train_dx_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",
    "train_dy_path": "/disk/scratch1/s1847450/data/conll09/bitgan/verb.bit",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",

    "model": {
        "autoencoder": {
            "type": "srl_ae_hub",
            "kl_alpha": 0.0,
            "ll_alpha": 1.0,
            //"decoder": {
            //    "type": "srl_graph_decoder",
            //    "input_dim": 20,  // predicate + label + (lemmas + roles),
            //    "dense_layer_dims": [4, 6],
            //    "dropout": 0.3,
            //},
            "decoder": {
                "type": "srl_lstms_decoder",
                "input_dim": 8,  // (300) predicate + label + last or (400) + context,
                "hidden_dim": 3, //  
                "always_use_predt": true,
                "dense_layer_dims": [4, 6],
                "dropout": 0.1,
            },
            "sampler": {
                "type": "uniform",
            }
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
            "token_dropout": 0.1,
            "lemma_dropout": 0.1,
            "label_dropout": 0.1,
            "predt_dropout": 0.1,
            "embed_lemma_ctx": true,
            "suppress_nonarg": true,
        },

        "type": "srl_vae_hub",
        "n_sample": 1,
        "ll_alpha": 0.5,
        "reweight": false, 
        "kl_prior": "null",
        "straight_through": true,
        "continuous_label": false,
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 4 
    },
    "trainer": {
        "type": "sri_vae",
        "num_epochs": 2,
        "grad_clipping": 1.0,
        "patience": 20,
        "shuffle": false,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": -1,
        "noun_loss_scalar": 1.0,
        "verb_loss_scalar": 1.0,
        "gen_skip_nepoch": 0,
        "gen_pretraining": -1,  
        "gen_loss_scalar": 1.0,
        "kld_loss_scalar": 0.5,
        "kld_update_rate": 0.05,
        "kld_update_unit": 5,
        "bpr_loss_scalar": 1.0,
        "sort_by_length": false,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        },
    }
}
