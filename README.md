
# A PyTorch Project Template

[![CodeFactor](https://www.codefactor.io/repository/github/chengzegang/pytorchtemplate/badge/master)](https://www.codefactor.io/repository/github/chengzegang/pytorchtemplate/overview/master)
[![test](https://github.com/chengzegang/PyTorchTemplate/actions/workflows/test.yml/badge.svg)](https://github.com/chengzegang/PyTorchTemplate/actions/workflows/test.yml)
[![build](https://github.com/chengzegang/PyTorchTemplate/actions/workflows/build.yml/badge.svg)](https://github.com/chengzegang/PyTorchTemplate/actions/workflows/build.yml)

This is a simple template for any research projects that require training deep learning models with PyTorch. It is based on my own everyday routine experiences. Originally, most functionalities were nested in some big singletons for the sake of simplicity (that states can be easily accessed from anywhere). However, as a project grows, it becomes more than critical to involve the so-called "Continous integration/Development" (CI/CD) paradigms for readability, testability, and maintainability. After all, it sucks when I need to debug every single function every fxxking time new things came in, especially considering the crazy fast pace in the academic world. Thus, this repo is born, and it aims to be more extensible and modularized, rather than a simple trainer python script.

-----

- [Introduction](#introduction)

- [License](#license)

## Introduction

The template is in flat-layout, and it is designed to be runnable without using ```pip install -e .```. There was a (many) time that I want to follow the Python Package Guide and tried to write everything in "src-layout". However, when someone submits their jobs in a cluster or any high-performance computing environment, it is very likely that they will not have permission to install a random custom package, OR, they will be required to install the packages in a container that is then locked when one script is running (Therefore you cannot run another project or change current environment until the current script is done). Thus, the flat-layout is the better choice for a research-oriented project.

The template contains four main submodules:

- datasets (all dataset creations and intializations)
- models (all model creations and intializations)
- trainers (**the main experiment logic happens here**)
- transforms (preprocessing, postprocessing, augmentation, etc.)

some helper modules:

- ddp.py (a helper module for distributed training)
- lr_lambda.py (a helper module for learning rate scheduling; since in most case "cosine warmup" is the only thing I need, I wrote them in one file)

some configs and inis:

- pyproject.toml (hatch configurations, it is always preferred to define all metadata in the pyproject.toml when possible)
- MANIFEST.in (manifest configurations, you may want to bundle these *.ini files or other data files into your compiled package)
- pytest.ini (pytest configurations, test is CRITIAL)
- mypy.ini (mypy configurations, python is not static-typed, but enforcing it will greatly reduce bugs in runtime)
- tox.ini (tox configurations, since we got many things to run and a nature programmer seeks to automate everything)
- requirements.txt (packages that are required to run the codes, and I don't think cpu version torch is anything related to deep learning)
- .flake8 (flake8 configurations for linting)
- .github/workflows/** (all github actions, you may want to test your codes before sync to github, but this one is only for the badges in README.md)
- .vscode/** (all vscode settings, personal preferences)

and a experimental extension module:

- extensions/rust_ext/** (a rust extension for Python, my personally interests in trying some rust codes)

It is true that one can abstract trainers into even smaller components, but being over-nerdy will only make the codes uneditable. It is more common in research projects that someone copies and pastes the codes of one ```trainer.py``` to be the skeleton of another. Moreover, this is also emphasized by the idea of "isolations": You really don't want your two experiments to interfere with each other. Thus, redundancy is somewhat encouraged and welcomed in trainer scripts (even though I still split the codes a little bit for further investigations).

## License

`project` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
