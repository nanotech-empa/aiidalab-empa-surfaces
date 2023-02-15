[![DOI](https://zenodo.org/badge/136625855.svg)](https://zenodo.org/badge/latestdoi/136625855)

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
