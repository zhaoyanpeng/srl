{
    "vocabulary": {
        "tokens_to_add": {"lemmas": ["NULL_LEMMA"], "tokens": ["NULL_TOKEN"]}
    },
    "dataset_reader":{
        "type":"conll2009",
        "feature_labels": ["pos", "dep"],
        "move_preposition_head": true,
        "max_num_argument": 7,
        "instance_type": "srl_graph"
    },
  
    "reader_mode": "srl_gan",

    //"train_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",
    //"test_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",

    "train_dx_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",
    "train_dy_path": "/disk/scratch1/s1847450/data/conll09/bitgan/verb.bit",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",

    "model": {
        "autoencoder": {
            "type": "srl_lemma_ae",
            "decoder": {
                //"type": "srl_graph_decoder",
                "type": "srl_basic_decoder",
                //"input_dim": 7, // 3 + 2 + 2
                "input_dim": 4, // 2 + 2
                "dense_layer_dims": [5, 5],
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

        //"type": "srl_vae_lhood",
        "type": "srl_vae_facto",
        "alpha": 0.5,
        "nsampling": 2,
        "coupled_loss": true,
        "straight_through": true,
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 3 
    },
    "trainer": {
        "type": "sri_vae_lemma",
        "num_epochs": 1,
        "grad_clipping": 1.0,
        "patience": 20,
        "shuffle": false,
        //"validation_metric": "+f1-measure-overall",
        "validation_metric": "-loss",
        "cuda_device": -1,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        },
    }
}
