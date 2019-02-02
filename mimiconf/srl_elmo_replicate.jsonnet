// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "dataset_reader":{
    "type":"conll2009",
    "feature_labels": ["pos", "dep"],
    "move_preposition_head": true, 
    "token_indexers": {
        "elmo": {
            "type": "elmo_characters"
        }
    }
    },
  
  //"train_data_path": "/disk/scratch1/s1847450/data/conll09/CoNLL2009-ST-English/CoNLL2009-ST-English-train.txt",
  //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt",
  //"test_data_path": "/disk/scratch1/s1847450/data/conll09/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt",
  
  //"train_data_path": "/disk/scratch1/s1847450/data/conll09/separated/train.noun",
  //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.noun",
  //"test_data_path": "/disk/scratch1/s1847450/data/conll09/separated/test.noun",
  
  "train_data_path": "/disk/scratch1/s1847450/data/conll09/separated/train.verb",
  "validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.verb",
  "test_data_path": "/disk/scratch1/s1847450/data/conll09/separated/test.verb",
  
  "vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src",
  "datasets_for_vocab_creation": ["vocab"],
  "model": {
    "type": "srl_naive",
    "embedding_dropout": 0.3,
    "text_field_embedder": {
      "token_embedders": {
        //"tokens": {
        //    "type": "embedding",
        //    "embedding_dim": 100,
        //    "pretrained_file": "/disk/scratch1/s1847450/data/embeding/glove.6B.100d.vec.c2009",
        //    "trainable": true 
        //}
        "elmo": {
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0.1
        }
      }
    },
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ]
    ],
    "encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 1124, 
      "hidden_size": 600,
      "num_layers": 3,
      "recurrent_dropout_probability": 0.3,
      "use_highway": true
    },
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
    "batch_size": 128 
  },
  //"evaluate_on_test": true,
  "trainer": {
    "type": "srl_naive",
    "num_epochs": 500,
    "grad_clipping": 1.0,
    "patience": 20,
    "num_serialized_models_to_keep": 10,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 3,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
