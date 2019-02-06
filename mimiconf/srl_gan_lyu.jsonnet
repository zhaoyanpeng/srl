// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "dataset_reader":{
    "type":"conll2009",
    "feature_labels": ["pos", "dep"],
    "move_preposition_head": true,
    "instance_type": "srl_gan",
    "token_indexers": {
        "elmo": {
            "type": "elmo_characters"
        }
    }
  },
  "reader_mode": "srl_gan",
  "dis_param_name": "srl_encoder",
  
  "train_dx_path": "/disk/scratch1/s1847450/data/conll09/separated/train.noun",
  "train_dy_path": "/disk/scratch1/s1847450/data/conll09/separated/train.verb",
  "validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.noun",
  "vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src",
  "datasets_for_vocab_creation": ["vocab"],
  "model": {
    "type": "srl_gan",
    "token_embedder": {
      "token_embedders": {
        //"tokens": {
        //    "type": "embedding",
        //    "embedding_dim": 2,
        //    //"pretrained_file": "/disk/scratch1/s1847450/data/embeding/glove.6B.100d.vec.c2009",
        //    "vocab_namespace": "tokens",
        //    "trainable": true 
        //}
        "elmo": {
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            //"options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
            //"weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0
        }
      }
    },
    "lemma_embedder": {
      "token_embedders": {
        "lemmas": {
            "type": "embedding",
            "embedding_dim": 300,
            "pretrained_file": "/disk/scratch1/s1847450/data/lemmata/en.lemma.300.20.vec.c2009",
            "vocab_namespace": "lemmas",
            "trainable": true 
        }
      }
    },
    "label_embedder": {
      "embedding_dim": 100,
      "vocab_namespace": "srl_tags",
      "trainable": true,
      "sparse": true 
    },
    "predicate_embedder": {
      "embedding_dim": 100,
      "vocab_namespace": "predicates",
      "trainable": true, 
      "sparse": true 
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
      "type": "srl_gan_dis",
      "embedding_dim": 500,
      "projected_dim": 200 
    },
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ]
    ],
    "binary_feature_dim": 100, 
    "regularizer": [
        [
            ".*scalar_parameters.*",
            {
                "type": "l2",
                "alpha": 0.001
            }
        ]
    ]
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 45 
  },
  "trainer": {
    "type": "srl_gan",
    "num_epochs": 500,
    "grad_norm": 5.0,
    //"grad_clipping": 1.0,
    "patience": 30,
    "shuffle": true,
    "num_serialized_models_to_keep": 10,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 1,
    "dis_skip_nepoch": 1,
    "gen_pretraining": 1,  
    "optimizer": {
      "type": "dense_sparse_adam",
      "betas": [0.9, 0.9]
    },
    "optimizer_dis": {
      "type": "dense_sparse_adam",
      "betas": [0.9, 0.9]
    }
  }
}
