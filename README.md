
# Transfer Learning of Semantic Role Models

Code for the paper *[Unsupervised Transfer of Semantic Role Models from Verbal to Nominal Domain](https://arxiv.org/pdf/2005.00278.pdf)* by [Yanpeng Zhao]() and [Ivan Titov](http://ivan-titov.org).

<p align="center"><img src="https://drive.google.com/uc?id=1wV0D9BGvVYqsNWPAW3vuteB4DkpwjRMR" alt="Transfer Learning of Semantic Role Labelers" width="50%"/></p>

Our model transfers the argument roles of verbal predicates (*acquired*) to the arguments of their nominalization (*acquisition*). *acquired* and *acquisition* share the lemma. Note that we do not rely on any argument alignment.

## Data

To be added...

## Dependencies

The SRL development environment relies on a specific commit of AllenNLP and needs a patch to fix a compatibility issue of that commit.

### Create Running Environment

```shell
git clone --branch beta https://github.com/zhaoyanpeng/srl.git
cd srl
virtualenv -p python3.7 ./pyenv/oops
source ./pyenv/oops/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install -r requirements.txt
```

### Install AllenNLP from Source

**ALERT:** this is an outdated repo. Before installing AllenNLP, maybe first resolve the mentioned issue (see *Resolve the AllenNLP Issue* section); unfortunately, I forget what exactly the issue is. 

```bash
git clone https://github.com/allenai/allennlp.git
cd allennlp
INSTALL_TEST_REQUIREMENTS=true scripts/install_requirements.sh
pip install --editable .
cd ..
```
### Resolve the AllenNLP Issue

```bash
git checkout 11d8327 -b patch
git checkout 385e66e -b srl_base
git checkout patch -- allennlp/semparse/domain_languages/domain_language.py
git branch -d patch
```


##  Running scripts

```bash
./executor/srl_toy.sh
```

## Citation
```bibtex
@misc{zhao2020unsupervised,
      title={Unsupervised Transfer of Semantic Role Models from Verbal to Nominal Domain}, 
      author={Yanpeng Zhao and Ivan Titov},
      year={2020},
      eprint={2005.00278},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://doi.org/10.48550/arXiv.2005.00278}
}
```

## License
MIT
