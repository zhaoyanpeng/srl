#!/usr/bin/env python
import logging
import os
import sys

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

from allennlp.commands import main  # pylint: disable=wrong-import-position
from nlpmimic.commands.srler import Srler
from nlpmimic.commands.boost import Boost
from nlpmimic.commands.archive import Archive
from nlpmimic.commands.srler_nyt import SrlerNyt 


if __name__ == "__main__":
    subcommand_overrides = {
            "srler": Srler(),
            "boost": Boost(),
            "archive": Archive(),
            "srler_nyt": SrlerNyt()
    }
    
    main(prog="allennlp", subcommand_overrides = subcommand_overrides)
