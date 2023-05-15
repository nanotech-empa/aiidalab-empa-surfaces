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

[![DOI](https://zenodo.org/badge/136625855.svg)](https://zenodo.org/badge/latestdoi/136625855)
