// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "dataset_reader":{
    "type":"conll2009",
    "feature_labels": ["pos", "dep"],
    "move_preposition_head": true 
    },
  "train_data_path": "/disk/scratch1/s1847450/data/conll09/separated/train.verb",
  "validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.verb",
  "test_data_path": "/disk/scratch1/s1847450/data/conll09/separated/test.verb",
  "vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src",
  "datasets_for_vocab_creation": ["vocab"],
  "model": {
    "type": "srl_naive",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "pretrained_file": "/disk/scratch1/s1847450/data/embeding/glove.6B.100d.vec.c2009",
            "trainable": true 
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
      "input_size": 200,
      "hidden_size": 300,
      "num_layers": 8,
      "recurrent_dropout_probability": 0.1,
      "use_highway": true
    },
    "binary_feature_dim": 100
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 128 
  },

  "trainer": {
    "type": "srl_naive",
    "num_epochs": 500,
    "grad_clipping": 1.0,
    "patience": 20,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 1,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
