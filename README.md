
The SRL development environment relies on a specific commit of Allennlp, and needs a patch to fix the compatibility problem of that commit.

## Installing AllenNLP from source

```bash
git clone https://github.com/allenai/allennlp.git
INSTALL_TEST_REQUIREMENTS=true scripts/install_requirements.sh
pip install --editable .
```
## Creating SRL development environment

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