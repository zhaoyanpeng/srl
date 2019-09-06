"""
The ``predict`` subcommand allows you to make bulk JSON-to-JSON
or dataset to JSON predictions using a trained model and its
:class:`~allennlp.service.predictors.predictor.Predictor` wrapper.

.. code-block:: bash

    $ allennlp predict -h
    usage: allennlp predict [-h] [--output-file OUTPUT_FILE]
                            [--weights-file WEIGHTS_FILE]
                            [--batch-size BATCH_SIZE] [--silent]
                            [--cuda-device CUDA_DEVICE] [--use-dataset-reader]
                            [-o OVERRIDES] [--predictor PREDICTOR]
                            [--include-package INCLUDE_PACKAGE]
                            archive_file input_file

    Run the specified model against a JSON-lines input file.

    positional arguments:
    archive_file          the archived model to make predictions with
    input_file            path to input file

    optional arguments:
    -h, --help              show this help message and exit
    --output-file OUTPUT_FILE
                            path to output file
    --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
    --batch-size BATCH_SIZE The batch size to use for processing
    --silent                do not print output to stdout
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
    -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
    --predictor PREDICTOR   optionally specify a specific predictor to use
    --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
from typing import List, Iterator, Optional
import argparse
import sys
import json

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance

from nlpmimic.data.dataset_readers.conll2009 import Conll2009DatasetReader, Conll2009Sentence

class Srler(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
        subparser.add_argument('input_file', type=str, help='path to input file')

        subparser.add_argument('--output-file', type=str, help='path to output file')
        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')

        subparser.add_argument('--silent', action='store_true', help='do not print output to stdout')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--predictor',
                               type=str,
                               help='optionally specify a specific predictor to use')

        subparser.set_defaults(func=_predict)

        return subparser

def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_file,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)

    return Predictor.from_archive(archive, args.predictor)


class _PredictManager:

    def __init__(self,
                 predictor: Predictor,
                 input_file: str,
                 output_file: str,
                 batch_size: int,
                 print_to_console: bool) -> None:

        self._predictor = predictor
        self._input_file = input_file

        gold_output_file = output_file + '.gold'
        pred_output_file = output_file + '.pred'
        
        self._gold_output_file = open(gold_output_file, 'w')
        self._pred_output_file = open(pred_output_file, 'w')

        self._batch_size = batch_size
        self._print_to_console = print_to_console
        
        self._dataset_reader = predictor._dataset_reader # pylint: disable=protected-access
        """
        self._dataset_reader = Conll2009DatasetReader(
                                token_indexers = {'elmo': ELMoTokenCharactersIndexer(namespace='elmo')},
                                valid_srl_labels = ["A0", "A1", "A2", "A3", "A4", "A5", 
                                    "AM-ADV", "AM-CAU", "AM-DIR", "AM-EXT", "AM-LOC", 
                                    "AM-MNR", "AM-NEG", "AM-PRD", "AM-TMP"],
                                lemma_file = "/disk/scratch1/s1847450/data/conll09/morph.only/all.morph.only.moved.arg.vocab.json",
                                lemma_use_firstk = 20,
                                feature_labels = ["pos", "dep"],
                                moved_preposition_head = ["IN", "TO"],
                                flatten_number = True,
                                max_num_argument = 7,
                                instance_type = "srl_graph",
                                allow_null_predicate = True) # pylint: disable=protected-access
        """

    def _prediction_to_str(self, output_dict: JsonDict, restore_head: bool = True) -> str:
        tokens = output_dict["tokens"]
        lemmas = output_dict["lemmas"]
        labels = output_dict["srl_tags"]
        pos_tags = output_dict["pos_tags"]
        head_ids = output_dict["head_ids"]
        predicate = output_dict["predicate"]
        predicate_index = output_dict["predicate_index"]
        
        prediction = Conll2009Sentence.initialize(predicate_index,
                                                  tokens,
                                                  labels,
                                                  lemmas,
                                                  pos_tags,
                                                  head_ids,
                                                  predicate)                                           
        if restore_head:
            for pos in self._dataset_reader.moved_preposition_head:
                prediction.restore_preposition_head_general(empty_label='_', preposition=pos)
        return prediction.format()
    
    def _predict_instances(self, 
                           batch_data: List[Instance], 
                           restore_head: bool = True) -> Iterator[str]:    
        if len(batch_data) == 1:
            results = [self._predictor.predict_instance(batch_data[0])]
        else:
            results = self._predictor.predict_batch_instance(batch_data)
        for output in results:
            yield self._prediction_to_str(output, restore_head)

    def _maybe_print_to_console_and_file(self,
                                         prediction: str,
                                         model_input: str = None) -> None:
        """ Not being used currently.
        """
        if self._print_to_console:
            if model_input is not None:
                print("input: ", model_input)
            print("prediction: ", prediction)

    def _get_instance_data(self) -> Iterator[Instance]:
        if self._input_file == "-":
            self.close()
            raise ConfigurationError("stdin is not an option when using a DatasetReader.")
        elif self._dataset_reader is None:
            self.close()
            raise ConfigurationError("To generate instances directly, pass a DatasetReader.")
        else:
            yield from self._dataset_reader.read(self._input_file)
    
    def create_gold_input(self, move_head: bool = False) -> None:
        has_reader = self._dataset_reader is not None
        if has_reader:
            self._dataset_reader.moved_preposition_head = ['IN', 'TO'] if move_head else []
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for gold_instance in batch:
                    gold_input = Conll2009Sentence.instance(gold_instance)
                    self._gold_output_file.write(gold_input.format() + '\n')
        else:
            raise ConfigurationError("Could you please provide a proper dataset reader?")
        print('move_head: {}'.format(self._dataset_reader.moved_preposition_head))

    def run(self, move_head: bool = True, restore_head: bool = True) -> None:
        has_reader = self._dataset_reader is not None
        if has_reader:
            self._dataset_reader.moved_preposition_head = ['IN', 'TO'] if move_head else [] 
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for gold_instance, result in zip(batch, self._predict_instances(batch, restore_head)):
                    #gold_input = Conll2009Sentence.instance(gold_instance)
                    #if restore_head:
                    #    gold_input.restore_preposition_head('_')
                    #self._gold_output_file.write(gold_input.format() + '\n')
                    self._pred_output_file.write(result + '\n')
            print('move_head: {}'.format(self._dataset_reader.moved_preposition_head))
        else:
            raise ConfigurationError("Could you please provide a proper dataset reader?")
    
    def close(self) -> None:
        if self._gold_output_file is not None:
            self._gold_output_file.close()
        if self._pred_output_file is not None:
            self._pred_output_file.close()

def _predict(args: argparse.Namespace) -> None:
    predictor = _get_predictor(args)

    if args.silent or not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    manager = _PredictManager(predictor,
                              args.input_file,
                              args.output_file,
                              args.batch_size,
                              not args.silent)
    try:
        manager.create_gold_input(move_head=True)
        manager.run(move_head=True, restore_head=False)
    except Exception as e:
        manager.close()
        print('err encountered: ', e)

