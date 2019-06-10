{
    "vocabulary": {
        "tokens_to_add": {"lemmas": ["NULL_LEMMA"]}
    },
    "dataset_reader":{
        "type":"conll2009",
        //"maximum_length": 80,
        //"valid_srl_labels": ["A1", "A0", "A2", "AM-TMP", "A3", "AM-MNR", "AM-LOC", "A4"],
        "lemma_file":  "/disk/scratch1/s1847450/data/conll09/all.moved.arg.vocab",
        "lemma_use_firstk": 20,
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
    "validation_ontraining_data": false,

    "dis_param_name": ["classifier.seq_encoder",
                       "classifier.label_projection_layer",
                       "classifier.psign_embedder",
                       "classifier.token_embedder", 
                       "classifier.lemma_embedder", 
                       "classifier.predt_embedder", 
                       "autoencoder.encoder",
                       "autoencoder.sampler"],

    "train_dx_path": "/disk/scratch1/s1847450/data/conll09/morph.word/5.0/train.noun",
    "train_dy_path": "/disk/scratch1/s1847450/data/conll09/morph.word/5.0/train.verb",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/morph.word/5.0/devel.noun",

    //"vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src",
    //"datasets_for_vocab_creation": ["vocab"],

    //"old_model_path": "/disk/scratch1/s1847450/model/srl_verb_lyu/best.th",
    //"update_key_set": {"text_field_embedder": "token_embedder", "encoder": "seq_encoder"},

    "model": {
        "autoencoder": {
            "type": "srl_graph_ae",
            "nsample": 2,
            "alpha": 0.5,
            "b_ctx_predicate": true,
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
            "decoder": {
                "type": "srl_lstms_decoder",
                "input_dim": 300, // predicate + label,
                "hidden_dim": 300, // 100 * 3 + 0  
                "always_use_predt": true,
                "dense_layer_dims": [450, 600],
                "b_ignore_z": false,
                "dropout": 0.1,
            },
            "sampler": {
                "type": "gaussian",
                "input_dim": 100, 
                "output_dim": 100,  
            },
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
            "embedding_dropout": 0.1,
            "suppress_nonarg": true,
        },

        "type": "srl_vae",
        "nsampling": 10,
        "alpha": 0.5,
        "reweight": true, 
        "straight_through": true,
        "continuous_label": false,
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 128 
    },
    "trainer": {
        "type": "sri_aggressive",
        "num_epochs": 1000,
        "grad_norm": 5.0,
        "grad_clipping": 1.0,
        "patience": 200,
        "shuffle": true,
        "num_serialized_models_to_keep": 1,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": 0,
        "dis_max_nbatch": 30,
        "dis_min_nbatch": 10,
        "aggressive_vae": true,
        "gen_skip_nepoch": 0,
        "gen_pretraining": -1, 
        "gen_loss_scalar": 1.0,
        "shuffle_arguments": false,
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
