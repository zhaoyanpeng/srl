{
    "vocabulary": {
        "tokens_to_add": {"predts": ["GLOBAL_PREDT"]}
    },
    "dataset_reader":{
        "type": "feature",
        "max_num_argument": 4,
    },

    //"train_data_path": "/disk/scratch1/s1847450/data/ivan15/train.txt",
    //"test_data_path": "/disk/scratch1/s1847450/data/ivan15/test.txt",
    //"validation_data_path": "/disk/scratch1/s1847450/data/ivan15/dev.txt",

    //"train_data_path": "/disk/scratch1/s1847450/data/ivan15/train17.txt",
    //"test_data_path": "/disk/scratch1/s1847450/data/ivan15/test17.txt",
    //"validation_data_path": "/disk/scratch1/s1847450/data/ivan15/dev17.txt",

    "train_data_path": "/disk/scratch1/s1847450/data/ivan15/train21.txt",
    "test_data_path": "/disk/scratch1/s1847450/data/ivan15/test21.txt",
    "validation_data_path": "/disk/scratch1/s1847450/data/ivan15/dev21.txt",

    "model": {
        "classifier": {
            "type": "srl_inf_feate",
            "feate_embedder": {
                "token_embedders": {
                    "feates": {
                        "type": "embedding",
                        "embedding_dim": 21, //17, //38,  
                        "vocab_namespace": "feates",
                        "trainable": true 
                    }
                }
            },
            "argmt_embedder": {
                "token_embedders": {
                    "argmts": {
                        "type": "embedding",
                        "embedding_dim": 100, //30,  
                        "vocab_namespace": "argmts",
                        "pretrained_file": "/disk/scratch1/s1847450/data/lemmata/en.lemma.100.20.vec.morph",
                        "trainable": false 
                    }
                }
            },
            "predt_embedder": {
                "token_embedders": {
                    "predts": {
                        "type": "embedding",
                        "embedding_dim": 105000, //85000, //25500, //7650, //17100, //9450, 
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
        "initializer": [
            [
                "classifier.predt_embedder.*weight",
                {
                    "type": "normal",
                    "mean": 0.1, 
                    "std": 0.31623, // sqrt(0.1) 
                }
            ]
        ],

        "type": "srl_vae_feate",
        "feature_dim": 50, 
        "unique_role": true,
        "loss_type": "ivan",
        "nsampling": 10,
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["arguments", "num_tokens"]],
        "batch_size": 100, 
    },
    "trainer": {
        "type": "sri_vae_feats",
        "num_epochs": 1000,
        "grad_norm": 5.0,
        "grad_clipping": 1.0,
        "patience": 100,
        "shuffle": false,
        "validation_metric": "+f1",
        //"validation_metric": "+f1-measure-overall",
        //"validation_metric": "-loss",
        "cuda_device": 2,
        "optimizer": {
            "type": "adagrad",
            //"lr": 1.0, 
            //"type": "adadelta",
            //"rho": 0.95
        },
    }
}
