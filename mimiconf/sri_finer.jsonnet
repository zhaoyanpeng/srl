{
    "vocabulary": {
        "tokens_to_add": {"lemmas": ["NULL_LEMMA"]}
    },
    "dataset_reader":{
        "type":"conll2009",
        "maximum_length": 80,
        //"valid_srl_labels": ["A0", "A1", "A2", "A3", "A4", "A5", "AA", 
        //                     "AM-ADV", "AM-CAU", "AM-DIR", "AM-DIS", "AM-EXT", "AM-LOC", "AM-MNR", 
        //                     "AM-MOD", "AM-NEG", "AM-PNC", "AM-PRD", "AM-PRT", "AM-REC", "AM-TMP"],
        "valid_srl_labels": ["A0", "A1", "A2", "A3", "A4", "A5", "AA", 
                             "AM-ADV", "AM-CAU",           "AM-DIS", "AM-EXT",  
                             "AM-MOD", "AM-NEG", "AM-PNC", "AM-PRD", "AM-PRT", "AM-REC"],
        "lemma_file":  "/disk/scratch1/s1847450/data/conll09/morph.only/verb.all.moved.arg.vocab",
        "lemma_use_firstk": 5,
        "predicate_file":  "/disk/scratch1/s1847450/data/conll09/morph.only/verb.all.predicate.vocab",
        "predicate_use_firstk": 20,
        "feature_labels": ["pos", "dep"],
        "moved_preposition_head": ["IN", "TO"],
        "flatten_number": true,
        "max_num_argument": 7,
        "instance_type": "srl_graph",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
  
    "reader_mode": "basic",
    "validation_ontraining_data": false,

    "train_data_path": "/disk/scratch1/s1847450/data/conll09/morph.only/train.verb",
    "test_data_path": "/disk/scratch1/s1847450/data/conll09/morph.only/test.verb",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/morph.only/devel.verb",

    "model": {
        "autoencoder": {
            "type": "srl_finer_ae",
            "kl_alpha": 0.0,
            "ll_alpha": 1.0,
            "re_alpha": 0.5,
            "re_type": "relu",
            "b_use_z": false,
            "b_ctx_predicate": false,
            "generative_loss": "maxmargin",
            "negative_sample": 10,
            //"encoder": {
            //    "type": "srl_graph_encoder",
            //    "input_dim": 100, 
            //    "layer_timesteps": [2, 2, 2, 2],
            //    "residual_connection_layers": {"2": [0], "3": [0, 1]},
            //    "dense_layer_dims": [100],
            //    "node_msg_dropout": 0.3,
            //    "residual_dropout": 0.3,
            //    "aggregation_type": "a",
            //    "combined_vectors": false,
            //},
            "decoder": {
                "type": "srl_graph_decoder",

                //"type": "srl_lstms_decoder",
                //"always_use_predt": true,
                //"hidden_dim": 300, //  

                "input_dim": 900,  // predicate + label (+ ctx),
                "dense_layer_dims": [750, 450],
                "dropout": 0.3,
            },
            "sampler": {
                "type": "uniform",

                //"type": "gumbel",
                //"tau_prior": 5,
                //"tau": 1,
            }
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
                "pretrained_file": "/disk/scratch1/s1847450/data/lemmata/en.lemma.100.20.vec.morph",
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
            "token_dropout": 0.3,
            "lemma_dropout": 0.0,
            "label_dropout": 0.3,
            "predt_dropout": 0.3,
            "metric_type": "weighted",
            "embed_lemma_ctx": true,
            "suppress_nonarg": true,
        },

        "type": "srl_vae_finer",
        "nsampling": 1,
        //"alpha": 0.5,
        "reweight": false, 
        "straight_through": true,
        "continuous_label": true,
        "way2relax_argmax": "softmax",
        "kl_prior": "null",
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 128 
    },
    "trainer": {
        "type": "sri_vae_finer",
        "num_epochs": 1000,
        "grad_norm": 5.0,
        "grad_clipping": 1.0,
        "patience": 200,
        "shuffle": true,
        "num_serialized_models_to_keep": 1,
        "validation_metric": "+f1",
        //"validation_metric": "+f1-measure-overall",
        "shuffle_arguments": false,
        "cuda_device": 1,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        },
    }
}
