
#pytest nlpmimic/tests/modules/nlpmimic_test.py  \
#    -s --show-capture=no -p no:warnings -vv 

pytest nlpmimic/tests/modules/seq2vec_encoders/sampler_test.py  \
    -s --show-capture=no -p no:warnings -vv 

#pytest nlpmimic/tests/data/dataset_readers/srl_graph_test.py  \
#    -s --show-capture=no -p no:warnings -vv 

#pytest nlpmimic/tests/data/dataset_readers/conllx_dataset_reader_test.py  \
#    -s --show-capture=no -p no:warnings -vv 

#pytest nlpmimic/tests/data/dataset_readers/srl_common_test.py \
#    -s --show-capture=no -p no:warnings -vv 

#pytest nlpmimic/tests/training/metrics/conll2009_scorer_test.py \
#    -s --show-capture=no -p no:warnings -vv 

#pytest nlpmimic/tests/training/metrics/dependency_based_f1_measure_test.py \
#    -s --show-capture=no -p no:warnings -vv 

#pytest nlpmimic/tests/data/dataset_readers/conll2009_dataset_reader_test.py \
#    -s --show-capture=no -p no:warnings -vv 
