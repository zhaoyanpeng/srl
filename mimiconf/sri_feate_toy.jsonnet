{
    "vocabulary": {
        "tokens_to_add": {"predts": ["GLOBAL_PREDT"]}
    },
    "dataset_reader":{
        "type": "feature",
        "lazy": false, 
        "max_num_argument": 7,
    },

    "train_data_path": "/disk/scratch1/s1847450/data/ivan15/train.toy.txt",
    //"train_data_path": "/disk/scratch1/s1847450/data/ivan15/train.txt",
    //"test_data_path": "/disk/scratch1/s1847450/data/ivan15/test.txt",
    "validation_data_path": "/disk/scratch1/s1847450/data/ivan15/train.toy.txt",

    "model": {
        "classifier": {
            "type": "srl_inf_feate",
            "feate_embedder": {
                "token_embedders": {
                    "feates": {
                        "type": "embedding",
                        "embedding_dim": 3,
                        "vocab_namespace": "feates",
                        "trainable": true 
                    }
                }
            },
            "argmt_embedder": {
                "token_embedders": {
                    "argmts": {
                        "type": "embedding",
                        "embedding_dim": 2,
                        "vocab_namespace": "argmts",
                        "trainable": true 
                    }
                }
            },
            "predt_embedder": {
                "token_embedders": {
                    "predts": {
                        "type": "embedding",
                        "embedding_dim": 18,
                        "vocab_namespace": "predts",
                        "trainable": true 
                    }
                }
            },
            "tau": 0.01,
            //"tunable_tau": false,
            //"psign_dim": 2,
            "seq_projection_dim": null,
            "feate_dropout": 0.0,
            "argmt_dropout": 0.0,
            "label_dropout": 0.0,
            "predt_dropout": 0.0,
            "metric_type": "clustering",
            "suppress_nonarg": false,
        },

        "type": "srl_vae_feate",
        "feature_dim": 3,
        "loss_type": "relu",
        //"nsampling": 1,
        //"straight_through": true,
        //"continuous_label": true,
        //"kl_prior": "null",
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["arguments", "num_tokens"]],
        "batch_size": 3 
    },
    "trainer": {
        "type": "sri_vae_feats",
        "num_epochs": 1,
        "grad_clipping": 1.0,
        "patience": 20,
        "shuffle": false,
        "validation_metric": "+f1-measure-overall",
        //"validation_metric": "-loss",
        "cuda_device": 1,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        },
    }
}
