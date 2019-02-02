// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "dataset_reader":{"type":"srl"},
  "train_data_path": "/disk/scratch1/s1847450/data/conll05/toy/train",
  //"train_data_path": "/disk/scratch1/s1847450/data/nombank/bit/train",
  //"validation_data_path": "/disk/scratch1/s1847450/data/nombank/bit/devel",
  //"train_data_path": "/disk/scratch1/s1847450/data/conll05/c12/train",
  //"validation_data_path": "/disk/scratch1/s1847450/data/conll05/c12/devel",
  //"train_data_path": "/disk/scratch/s1847450/data/ontonotes/conll-formatted-ontonotes-5.0/data/train",
  //"validation_data_path": "/disk/scratch/s1847450/data/ontonotes/conll-formatted-ontonotes-5.0/data/development",
  "model": {
    "type": "srl",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
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
      "type": "alternating_lstm",
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
    "batch_size" : 80
  },

  "trainer": {
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
