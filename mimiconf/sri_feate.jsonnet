{
    "vocabulary": {
        "tokens_to_add": {"predts": ["GLOBAL_PREDT"]}
    },
    "dataset_reader":{
        "type":"feature",
        "max_num_argument": 4,
    },

    //"train_data_path": "/disk/scratch1/s1847450/data/ivan15/train.txt",
    //"test_data_path": "/disk/scratch1/s1847450/data/ivan15/test.txt",
    //"validation_data_path": "/disk/scratch1/s1847450/data/ivan15/dev.txt",

    "train_data_path": "/disk/scratch1/s1847450/data/ivan15/train17.txt",
    "test_data_path": "/disk/scratch1/s1847450/data/ivan15/test17.txt",
    "validation_data_path": "/disk/scratch1/s1847450/data/ivan15/dev17.txt",

    //"train_data_path": "/disk/scratch1/s1847450/data/ivan15/train21.txt",
    //"test_data_path": "/disk/scratch1/s1847450/data/ivan15/test21.txt",
    //"validation_data_path": "/disk/scratch1/s1847450/data/ivan15/dev21.txt",

    "model": {
        "classifier": {
            "type": "srl_inf_feate",
            "feate_embedder": {
                "token_embedders": {
                    "feates": {
                        "type": "embedding",
                        "embedding_dim": 17,//21, // //38,
                        "vocab_namespace": "feates",
                        "trainable": true 
                    }
                }
            },
            "argmt_embedder": {
                "token_embedders": {
                    "argmts": {
                        "type": "embedding",
                        "embedding_dim": 30,
                        "vocab_namespace": "argmts",
                        "trainable": true 
                    }
                }
            },
            "predt_embedder": {
                "token_embedders": {
                    "predts": {
                        "type": "embedding",
                        "embedding_dim": 7650, //9450, // //17100,
                        "vocab_namespace": "predts",
                        "trainable": true 
                    }
                }
            },
            "tau": 0.01,
            "seq_projection_dim": null,
            "feate_dropout": 0.0,
            "argmt_dropout": 0.0,
            "label_dropout": 0.0,
            "predt_dropout": 0.0,
            "suppress_nonarg": false,
        },

        "type": "srl_vae_feate",
        "feature_dim": 15,
        "loss_type": "relu",
        "nsampling": 10,
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["arguments", "num_tokens"]],
        "batch_size": 128, 
    },
    "trainer": {
        "type": "sri_vae_feats",
        "num_epochs": 1000,
        "grad_norm": 5.0,
        "grad_clipping": 1.0,
        "patience": 100,
        "shuffle": false,
        "validation_metric": "+f1-measure-overall",
        //"validation_metric": "-loss",
        "cuda_device": 2,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        },
    }
}
