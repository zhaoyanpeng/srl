// Configuration for a semantic role labeler model based on:
//   He, Luheng et al. “Deep Semantic Role Labeling: What Works and What's Next.” ACL (2017).
{
  "vocabulary": {
    "tokens_to_add": {"lemmas": ["NULL_LEMMA"]}
  },
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
  //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/devel.small.noun",
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
            "trainable": false 
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
      "type": "srl_gan_dis",
      "module_choice": "c",
      "embedding_dim": 6,
      "projected_dim": 4,
      "hidden_size": 2,
      "attent_size": 2,
      "num_layer": 1
    },
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
          //"type": "xavier_uniform"
          //"type": "normal"
          //"type": "uniform"
          //"type": "xavier_normal"
        }
      ],
      //[
        //"forward_layer.*weight",
        //{
          //"type": "xavier_uniform"
        //}
      //],
      //[
        //"backward_layer.*weight",
        //{
          //"type": "xavier_uniform"
        //}
      //]
    ],
    "binary_feature_dim": 2, 
    "temperature": 0.01,
    "fixed_temperature": true,
    "mask_empty_labels": false,
    "zero_null_lemma_embedding": true,
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 1 
  },
  "trainer": {
    "type": "srl_gan",
    "num_epochs": 2,
    "grad_clipping": 1.0,
    "patience": 20,
    "shuffle": false,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 3,
    "dis_skip_nepoch": 0,
    "gen_pretraining": -1,  
    "dis_loss_scalar": 0.01,
    "gen_loss_scalar": 5.0,
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
