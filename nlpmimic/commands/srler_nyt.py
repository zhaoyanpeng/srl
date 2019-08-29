from typing import List, Any, Iterator, Optional
import argparse
import sys
import json

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

from nlpmimic.data.dataset_readers.conll2009 import Conll2009Sentence
from nlpmimic.data.dataset_readers.conllx_unlabeled import ConllxUnlabeledDatasetReader

class SrlerNyt(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')

        subparser.add_argument('context_file', type=str, help='path to context file')
        subparser.add_argument('appendix_file', type=str, help='path to appendix file')

        subparser.add_argument('--indxes-file', type=str, default=None, help='path to output file')
        subparser.add_argument('--output-file', type=str, help='path to output file')
        subparser.add_argument('--output-gold', type=bool, default=False, help='wether output gold file or not')

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




def concat(list_a: List[str], list_b: List[str]):
    return [e1 + e2 for e1, e2 in zip(list_a, list_b)]

def padding_with_space(items: List[Any], 
                       width: int = 0,
                       align_left: bool = True) -> List[str]: 
    return ['{: <{}}'.format(x, width) for x in items] if align_left \
        else ['{: >{}}'.format(x, width) for x in items]

def padding_with_tab(items: List[Any], 
                     align_left: bool = True) -> List[str]:
    return ['{}\t'.format(x) for x in items] if align_left \
        else ['\t{}'.format(x) for x in items]

def pretty_format(data, read_friendly: bool = True, space_width: int = 2) -> str:
    """ FIXME: Here assumming a valid Conll 2009 sentence, add sanity check.
    """
    # format string items
    columns = []
    if not read_friendly:
        for idx, field in enumerate(data):
            field = ['_' if x is None else x for x in field] # remove None
            columns.append(padding_with_tab(field))
    else:
        for idx, field in enumerate(data):
            field = ['_' if x is None else x for x in field] # remove None
            width = max([0] + [len(str(x)) for x in field]) + space_width
            columns.append(padding_with_space(field, width, True))
    # one item
    if len(columns) < 2:
        return '\n'.join(columns[0]) + '\n' 

    # concatenate string items
    rows = concat(columns[0], columns[1])
    for field in columns[2:]:
        rows = concat(rows, field) 
    return '\n'.join(rows) + '\n' 




class _PredictManager:

    def __init__(self,
                 predictor: Predictor,
                 context_file: str,
                 appendix_file: str,
                 indxes_file: str,
                 output_file: str,
                 batch_size: int,
                 output_gold: bool,
                 print_to_console: bool) -> None:

        self._predictor = predictor
        self._context_file = context_file
        self._appendix_file = appendix_file

        pred_output_file = output_file + '.pred'
        self._pred_output_file = open(pred_output_file, 'w')
        
        if output_gold:
            gold_output_file = output_file + '.gold'
            self._gold_output_file = open(gold_output_file, 'w')
        else:
            self._gold_output_file = None
        
        if indxes_file:
            self._output_idxes = open(indxes_file, 'r')
        else:
            self._output_idxes = None

        self._batch_size = batch_size
        self._print_to_console = print_to_console
        
        self._dataset_reader = ConllxUnlabeledDatasetReader(
                                 token_indexers = {'elmo': ELMoTokenCharactersIndexer(namespace='elmo')},
                                 feature_labels = ['pos', 'dep'], 
                                 move_preposition_head = True,
                                 allow_null_predicate = True,
                                 instance_type = 'basic') # pylint: disable=protected-access

    def _prediction_to_str(self, output_dict: JsonDict, restore_head: bool = True) -> str:
        tokens = output_dict["tokens"]
        labels = [x if x != 'O' else '_' for x in output_dict["srl_tags"]]
        return tokens, labels 
    
    def _predict_instances(self, 
                           batch_data: List[Instance], 
                           restore_head: bool = True) -> Iterator[str]:    
        if len(batch_data) == 1:
            results = [self._predictor.predict_instance(batch_data[0])]
        else:
            results = self._predictor.predict_batch_instance(batch_data)
        for output in results:
            yield self._prediction_to_str(output, restore_head)
    
    def _write_instances(self, labels: List[list], cnt: int, def_end = 'ENDING_OF_FILE'):
        if len(labels) == 0:
            print('err: ', labels, cnt)
        while len(labels) > 0:
            ncol = next(self._output_idxes, def_end)
            if ncol == def_end:
                print('err: ', labels, cnt)
            end = int(ncol)
            
            data = labels[:end]
            if len(data) != end:
                print('err: ', labels, cnt)
            labels = labels[end:] 
            self._pred_output_file.write(pretty_format(data) + '\n')

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
        if self._context_file == "-":
            self.close()
            raise ConfigurationError("stdin is not an option when using a DatasetReader.")
        elif self._dataset_reader is None:
            self.close()
            raise ConfigurationError("To generate instances directly, pass a DatasetReader.")
        else:
            yield from self._dataset_reader._read(self._context_file, self._appendix_file)
    
    def run(self, restore_head: bool = True) -> None:
        has_reader = self._dataset_reader is not None
        if has_reader:
            cnt = 0
            empty = False 
            current_tokens = []
            current_labels = []
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for gold_instance, (tokens, labels) in zip(batch, self._predict_instances(batch, restore_head)):
                    if tokens != current_tokens:
                        if current_tokens != []:
                            if self._output_idxes is None:    
                                result = pretty_format(current_labels) 
                                self._pred_output_file.write(result + '\n')
                            else: 
                                cnt += 1
                                self._write_instances(current_labels, cnt)
                            
                            empty = True

                        current_tokens = tokens
                        current_labels = [labels]
                        empty = False
                    else:
                        current_labels.append(labels) 
                        empty = False
            if not empty:
                result = pretty_format(current_labels) 
                self._pred_output_file.write(result + '\n')
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
                              args.context_file,
                              args.appendix_file,
                              args.indxes_file,
                              args.output_file,
                              args.batch_size,
                              args.output_gold,
                              not args.silent)
    try:
        manager.run(restore_head=False)
    except Exception as e:
        manager.close()
        print('err encountered: ', e)

