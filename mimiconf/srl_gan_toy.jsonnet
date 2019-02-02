// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "dataset_reader":{
    "type":"conll2009",
    "feature_labels": ["pos", "dep"],
    "move_preposition_head": true,
    "instance_type": "srl_gan"
    },
  "reader_mode": "srl_gan",
  "dis_param_name": "srl_encoder",
  
  "train_dx_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",
  "train_dy_path": "/disk/scratch1/s1847450/data/conll09/bitgan/verb.bit",
  "validation_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",
  //"vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src",
  //"datasets_for_vocab_creation": ["vocab"],
  "model": {
    "type": "srl_gan",
    "token_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 2,
            //"pretrained_file": "/disk/scratch1/s1847450/data/embeding/glove.6B.100d.vec.c2009",
            "vocab_namespace": "tokens",
            "trainable": true 
        }
      }
    },
    "lemma_embedder": {
      "token_embedders": {
        "lemmas": {
            "type": "embedding",
            "embedding_dim": 2,
            //"pretrained_file": "/disk/scratch1/s1847450/data/embeding/glove.6B.100d.vec.c2009",
            "vocab_namespace": "lemmas",
            "trainable": true 
        }
      }
    },
    "label_embedder": {
      "embedding_dim": 2,
      "vocab_namespace": "srl_tags",
      "trainable": true,
      "sparse": false 
    },
    "predicate_embedder": {
      "embedding_dim": 2,
      "vocab_namespace": "predicates",
      "trainable": true, 
      "sparse": false 
    },
    "seq_encoder": {
      "type": "stacked_bidirectional_lstm",
      "input_size": 4,
      "hidden_size": 2,
      "num_layers": 1,
      "recurrent_dropout_probability": 0.0,
      "use_highway": true
    },
    "srl_encoder": {
      "type": "cnn",
      "embedding_dim": 4,
      "num_filters": 50,
      "output_dim": 1
    },
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ]
    ],
    "binary_feature_dim": 2 
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 2 
  },
  "trainer": {
    "type": "srl_gan",
    "num_epochs": 1,
    "grad_clipping": 1.0,
    "patience": 20,
    "shuffle": false,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 2,
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
