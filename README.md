[![DOI](https://zenodo.org/badge/110861368.svg)](https://zenodo.org/badge/latestdoi/110861368)

# AiiDAlab - Empa Surfaces app

The Empa Surfaces AiiDAlab app enables the user to prepare and run automatic AiiDA workflows for calculations relevant in on-surface chemistry.
Features include:

* generating various metal slab geometries;
* geometry optimizations of adsorbed systems, bulk systems and isolated molecules;
* chains of constrained geometry optimizations;
* Nudged Elastic Band (NEB) calculations;
* GW calculations for isolated systems;
* ...

## AiiDAlab app for Scanning Probe Microscopy simulations

This repository contains AiiDA workflows and Jupyter GUI for running scanning probe microscopy simulations.
The calculations are performed on top of a CP2K wave function optimization (ran with diagonalization) with the [cp2k-spm-tools](https://github.com/nanotech-empa/cp2k-spm-tools).
The external tools need to be set up on the remote computer and can be conveniently done with `setup_codes` interface, which also installs the corresponding AiiDA plugins.

Features include:

* Scanning tunnelling microscopy/spectroscopy (STM/STS) simulations.
* Orbital analysis - calculates the STM/STS signatures of orbitals of isolated molecules.
* Projected Density of States (PDOS) analysis - calculates the DOS projection on the adsorbed molecule together with gas-phase orbital content.
* Atomic force microscopy (AFM) simulations - calls the [ProbeParticle](https://github.com/ProkopHapala/ProbeParticleModel) code.
* High-resolution STM - simulates the STM/STS signatures performed with the CO tip.

## Running the tests

Install the development dependencies and run the suite with pytest:

```
pip install -e .[dev]
pytest
```

## For maintainers

To create a new release, clone the repository, install development dependencies with `pip install '.[dev]'`, and then execute `bumpver update --dry --major (--minor/--patch)`.
This will display the changes that will be made to the repository - check them carefully.

Once you are happy with the changes, remove the `--dry` option and re-execute the command.
This will:

  1. Create a tagged release with bumped version and push it to the repository.
  2. Trigger a GitHub actions workflow that creates a GitHub release.

Additional notes:

  - The release tag (e.g. a/b/rc) is determined from the last release.
    Use the `--tag beta (alpha/gamma)`  option to switch the release tag.


## Contact

For inquiries concerning Materials Cloud, please use the [public issue tracker](https://github.com/materialscloud-org/issues).
For inquiries concerning the Empa Surfaces App, please contact [carlo.pignedoli@empa.ch](mailto:carlo.pignedoli@empa.ch).

## Acknowledgements
We acknowledge support from:
* the [NCCR MARVEL](http://nccr-marvel.ch/) funded by the Swiss National Science Foundation;
* the [swissuniversities P-5 project "Materials Cloud"](https://www.materialscloud.org/swissuniversities).

<img src="https://github.com/nanotech-empa/aiida-nanotech-empa/blob/master/images/MARVEL.png" width="250px" height="131px"/>
<img src="https://github.com/nanotech-empa/aiida-nanotech-empa/blob/master/images/swissuniversities.png" width="300px" height="35px"/>


Historical scanning-probe app (a separate software archive):

[![DOI](https://zenodo.org/badge/136625855.svg)](https://zenodo.org/badge/latestdoi/136625855)

## Software attribution and scientific acknowledgements

Please credit [aiidalab-empa-surfaces](https://github.com/nanotech-empa/aiidalab-empa-surfaces) and its
[contributors](https://github.com/nanotech-empa/aiidalab-empa-surfaces/graphs/contributors),
and identify the release or immutable commit used. Existing author,
contributor and funding acknowledgements remain authoritative.

See [CITATION.cff](CITATION.cff) for the software citation. A project DOI
covers archived releases; identify the exact version or commit separately.
Use a release DOI only for the release it identifies.

Preserve the copyright and permission notices in [LICENSE.txt](LICENSE.txt)
when copying or distributing copies or substantial portions of MIT-licensed
code. Separately licensed components retain their own notices. A name or
repository link does not replace the applicable licence notices.

For research using this app, acknowledge the [AiiDA team](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md#aiida)
and [AiiDAlab team](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md#aiidalab), and cite the three AiiDA papers
and both AiiDAlab platform papers listed in that guide.

Also credit aiida-nanotech-empa, cp2k-spm-tools, the simulation codes and the particular STM/STS, AFM or other methods used. The historical scanning-probe app retains its separate software credit.

The [shared credit guide](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md) keeps software attribution, licence
compliance and scientific citation requests separate. Scientific citation
requests are not additional MIT licence conditions.
