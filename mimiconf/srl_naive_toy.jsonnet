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
    "instance_type": "srl_graph"
    },
  "reader_mode": "srl_gan",
  "dis_param_name": ["srl_encoder", "predicate_embedder", "label_embedder"],
  
  //"train_dx_path": "/disk/scratch1/s1847450/data/conll09/separated/noun.morph.picked",
  //"train_dy_path": "/disk/scratch1/s1847450/data/conll09/separated/verb.morph.picked",
  "train_dx_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",
  "train_dy_path": "/disk/scratch1/s1847450/data/conll09/bitgan/verb.bit",
  "validation_data_path": "/disk/scratch1/s1847450/data/conll09/bitgan/noun.bit",
  //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/devel.small.noun",
  //"vocab_src_path": "/disk/scratch1/s1847450/data/conll09/separated/vocab.src",
  //"datasets_for_vocab_creation": ["vocab"],
  "model": {
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
      "type": "srl_naive_dis",
      "module_choice": "c",
      "embedding_dim": 6,
      //"embedding_dim": 15,
      "projected_dim": 4,
      "hidden_size": 2,
      "attent_size": 2,
      "num_layer": 1,
      "num_model": 0 

      //"type": "srl_graph_dis",
      //"layer_timesteps": [2, 2, 2, 2],
      //"residual_connection_layers": {"2": [0], "3": [0, 1]},
      //"node_msg_dropout": 0.3,
      //"residual_dropout": 0.3,
      //"aggregation_type": "c",
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
    "type": "srl_gan",
    "binary_feature_dim": 2, 
    "temperature": 0.01,
    "fixed_temperature": false,
    "mask_empty_labels": false,
    //"use_label_indicator": true,
    "zero_null_lemma_embedding": true,
    
    "label_loss_type": "unscale_kl",
    "regularized_labels": ["O"],
    "regularized_nonarg": true,
    "regularized_batch": true,
    //"suppress_nonarg": true,
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 5 
  },
  "trainer": {
    "type": "srl_gan",
    "num_epochs": 5,
    "grad_clipping": 1.0,
    "patience": 20,
    "shuffle": false,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 3,
    "dis_min_loss": 0.45,
    "dis_skip_nepoch": 0,
    "gen_skip_nepoch": 0,
    "gen_pretraining": -1,  
    "dis_loss_scalar": 0.05,
    "gen_loss_scalar": 1.0,
    "kld_loss_scalar": 0.5,
    "sort_by_length": true,
    "consecutive_update": false,
    "dis_max_nbatch": 2,
    "gen_max_nbatch": 4,
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
