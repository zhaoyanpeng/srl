{
    "dataset_reader":{
        "type":"conll2009_bare",
        "move_preposition_head": false,
    },
  
    "train_data_path": "/disk/scratch1/s1847450/data/conll09/separated/train.verb",
    //"test_data_path": "/disk/scratch1/s1847450/data/conll09/separated/test.verb",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.verb",

    //"train_data_path": "/disk/scratch1/s1847450/data/conll09/separated/train.noun",
    //"test_data_path": "/disk/scratch1/s1847450/data/conll09/separated/test.noun",
    //"validation_data_path": "/disk/scratch1/s1847450/data/conll09/separated/devel.noun",

    "trainer": {
        "type": "sri_syntx",
    }
}
