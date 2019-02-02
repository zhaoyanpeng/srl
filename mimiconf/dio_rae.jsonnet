// Configuration for a constituency parser based on:
//   Stern, Mitchell et al. “A Minimal Span-Based Neural Constituency Parser.” ACL (2017).
{
    "dataset_reader":{
        "type":"penn_treebank",
        "use_pos_tags": true
    },
    "train_data_path": "/disk/scratch1/s1847450/data/Data.Prd/wsj.21",
    //"validation_data_path": std.extVar('PTB_DEV_PATH'),
    //"test_data_path": std.extVar('PTB_TEST_PATH'),
    "model": {
      "type": "dio_rae",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": 300,
            //"embedding_dim": 4,
            "pretrained_file": "/disk/scratch1/s1847450/data/embeding/glove.6B.300d.txt",
            "trainable": true
          }
        }
      },
      "encoder": {
        "type": "dio_lstm",
        "input_size": 100,
        "hidden_size": 100
      },
      "feedforward": {
        "input_dim": 300,
        "num_layers": 1,
        "hidden_dims": 200,
        "activations": "relu",
        "dropout": 0.1
      },
    },
    "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "num_tokens"]],
      "batch_size" : 1 
    },
    "trainer": {
      "type": "dio_rae",
      "learning_rate_scheduler": {
        "type": "multi_step",
        "milestones": [40, 50, 60, 70, 80],
        "gamma": 0.8
      },
      "model_save_interval": 28800, 
      "shuffle": true,
      "num_epochs": 100,
      "grad_norm": 5.0,
      "patience": 20,
      "validation_metric": "+evalb_f1_measure",
      "cuda_device": -1,
      "optimizer": {
        "type": "adadelta",
        "lr": 1.0,
        "rho": 0.95
      }
    }
  }

