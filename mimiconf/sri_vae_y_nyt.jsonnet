{
    "vocabulary": {
        "tokens_to_add": {"lemmas": ["NULL_LEMMA"]}
    },
    "devel_dataset_reader":{
        "type":"conll2009",
        "valid_srl_labels": ["A0", "A1", "A2", "A3", "A4", "A5", 
                             "AM-ADV", "AM-CAU", "AM-DIR", "AM-EXT", "AM-LOC", 
                             "AM-MNR", "AM-NEG", "AM-PRD", "AM-TMP"],
        "lemma_file": "/disk/scratch1/s1847450/data/conll09/morph.only/all.morph.only.moved.arg.vocab.json",
        "lemma_use_firstk": 20,
        //"predicate_file": "/disk/scratch1/s1847450/data/conll09/morph.only/train.noun.pred.vocab",
        //"predicate_use_firstk": 20,
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

    "train_dataset_reader":{
        "type":"conll2009",
        "maximum_length": 80,
        "valid_srl_labels": ["A0", "A1", "A2", "A3", "A4", "A5", 
                             "AM-ADV", "AM-CAU", "AM-DIR", "AM-EXT", "AM-LOC", 
                             "AM-MNR", "AM-NEG", "AM-PRD", "AM-TMP"],
        "lemma_file": "/disk/scratch1/s1847450/data/conll09/morph.only/all.morph.only.moved.arg.vocab.json",
        "lemma_use_firstk": 20,
        "predicate_file": "/disk/scratch1/s1847450/data/conll09/morph.only/train.noun.pred.vocab",
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

    "nytimes_reader": {
        "type":"conllx_unlabeled",
        "maximum_length": 80,
        "valid_srl_labels": ["A0", "A1", "A2", "A3", "A4", "A5", 
                             "AM-ADV", "AM-CAU", "AM-DIR", "AM-EXT", "AM-LOC", 
                             "AM-MNR", "AM-NEG", "AM-PRD", "AM-TMP"],
        "lemma_file": "/disk/scratch1/s1847450/data/conll09/morph.only/all.morph.only.moved.arg.vocab.json",
        "lemma_use_firstk": 20,
        "predicate_file": "/disk/scratch1/s1847450/data/conll09/morph.only/train.noun.pred.vocab",
        "predicate_use_firstk": 10,
        "feature_labels": ["pos", "dep"],
        "flatten_number": true,
        "max_num_argument": 7,
        "instance_type": "srl_graph",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
  
  
    "reader_mode": "srl_nyt",
    "validation_ontraining_data": false,

    //"train_dx_path": "/disk/scratch1/s1847450/data/conll09/morph.word/5.0/train.noun",
    //"train_dy_path": "/disk/scratch1/s1847450/data/conll09/morph.word/5.0/train.verb",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/morph.only/devel.noun.morph.only",

    "train_dx_path": "/disk/scratch1/s1847450/data/conll09/morph.word/n20.0/train.noun",
    "train_dy_path": "/disk/scratch1/s1847450/data/conll09/morph.word/n20.0/train.verb",
    "validation_data_path": "/disk/scratch1/s1847450/data/conll09/morph.only/devel.noun.morph.only",

    "train_dy_context_path":  "/disk/scratch1/s1847450/data/nytimes.new/morph.word/nyt.verb.20.10.1000",

    //"train_dx_path": "/disk/scratch1/s1847450/data/conll09/morph.only/train.noun.morph.only",
    //"train_dy_path": "/disk/scratch1/s1847450/data/conll09/morph.only/train.verb.morph.only",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/morph.only/devel.noun.morph.only",

    //"vocab_src_path": "/disk/scratch1/s1847450/data/conll09/morph.only/all.morph.only.vocab.src",
    //"datasets_for_vocab_creation": ["vocab"],

    //"old_model_path": "/disk/scratch1/s1847450/model/sri_lhood_v20.0/best.th",
    //"update_key_set": {"classifier": "classifier", "autoencoder": "autoencoder"},
    //"tunable": false,

    "model": {
        "autoencoder": {
            "type": "srl_lstms_ae",
            "kl_alpha": 0.0,
            "ll_alpha": 1.0,
            "b_use_z": false,
            "b_ctx_predicate": false,
            "decoder": {
                "type": "srl_graph_decoder",
                "input_dim": 900, // (200)z + predicate + label or (900) ctx lemmas,
                "dense_layer_dims": [450, 600],
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
                    "elmo": {
                        "type": "elmo_token_embedder",
"options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
"weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
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
            "token_dropout": 0.1,
            "lemma_dropout": 0.1,
            "label_dropout": 0.1,
            "predt_dropout": 0.1,
            "embed_lemma_ctx": true,
            "suppress_nonarg": true,
        },

        "type": "srl_vae_d",
        "nsampling": 10,
        "alpha": 0.5,
        "reweight": true, 
        "straight_through": true,
        "continuous_label": false,
        "kl_prior": null,
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 128 
    },
    "trainer": {
        "type": "sri_vae",
        "num_epochs": 1000,
        "grad_clipping": 1.0,
        "patience": 200,
        "shuffle": true,
        "num_serialized_models_to_keep": 1,
        "validation_metric": "+f1-measure-overall",
        "cuda_device": 3,
        "noun_loss_scalar": 1.0,
        "verb_loss_scalar": 1.0,
        "gen_skip_nepoch": 0,
        "gen_pretraining": -1, 
        "gen_loss_scalar": 1.0,
        "shuffle_arguments": false,
        "optimizer": {
            "type": "adadelta",
            "rho": 0.95
        },
    }
}
