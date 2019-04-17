// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "vocabulary": {
    "tokens_to_add": {"lemmas": ["NULL_LEMMA"]}
  },
  "dataset_reader": {
    "type":"conll2009",
    //"maximum_length": 80,
    //"valid_srl_labels": ["A1", "A0", "A2", "AM-TMP", "A3", "AM-MNR", "AM-LOC", "AM-EXT", "AM-NEG", "AM-ADV", "A4"],
    "feature_labels": ["pos", "dep"],
    "move_preposition_head": true,
    "instance_type": "srl_graph",
    "token_indexers": {
        "elmo": {
            "type": "elmo_characters"
        }
    }
  },
  "reader_mode": "srl_gan",
  //"dis_param_name": ["srl_encoder", "predicate_embedder"],
  "dis_param_name": ["srl_encoder", "predicate_embedder", "lemma_embedder"],
  
  "train_dx_path": "/disk/scratch1/s1847450/data/conll09/separated/noun.morph.only.sel",
  "train_dy_path": "/disk/scratch1/s1847450/data/conll09/separated/verb.morph.only.sel",
  "validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.noun.sel",

  "vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src",
  "datasets_for_vocab_creation": ["vocab"],
  "model": {
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
            "pretrained_file": "/disk/scratch1/s1847450/data/lemmata/en.lemma.100.20.vec.sells",
            "vocab_namespace": "lemmas",
            "trainable": false 
        }
      }
    },
    "label_embedder": {
      // ignored in `graph` mode
      "embedding_dim": 300,
      "vocab_namespace": "srl_tags",
      "trainable": true,
      "sparse": false 
    },
    "predicate_embedder": {
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
    "srl_encoder": {
      "type": "srl_graph_dis",
      "layer_timesteps": [2, 2, 2, 2],
      "residual_connection_layers": {"2": [0], "3": [0, 1]},
      "node_msg_dropout": 0.3,
      "residual_dropout": 0.3,
      "aggregation_type": "c",
    },
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
            "type": "orthogonal"
        }
      ]
    ],
    "type": "srl_graph",
    "binary_feature_dim": 100, 
    "temperature": 1,
    "fixed_temperature": false,
    "mask_empty_labels": false, 
    //"use_label_indicator": true,
    "optimize_lemma_embedding": true,
    "zero_null_lemma_embedding": true,
    
    "label_loss_type": "unscale_kl",
    "regularized_batch": true,
    "regularized_labels": ["O"],
    "regularized_nonarg": false,
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 64,
    "padding_noise": 0.0
  },
  "trainer": {
    "type": "srl_gan",
    "num_epochs": 1000,
    "grad_clipping": 1.0,
    "patience": 150,
    "shuffle": true,
    "num_serialized_models_to_keep": 5,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 0,
    "dis_min_loss": 0.0,
    "dis_skip_nepoch": 0,
    "gen_skip_nepoch": 0,
    "gen_pretraining": -1, 
    "dis_loss_scalar": 0.05,
    "gen_loss_scalar": 1.0,
    "kld_loss_scalar": 0.5,
    "bpr_loss_scalar": 3.0,
    "sort_by_length": true,
    "consecutive_update": false,
    "dis_max_nbatch": 2,
    "gen_max_nbatch": 8,
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
