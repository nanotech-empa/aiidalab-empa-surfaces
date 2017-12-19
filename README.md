[![DOI](https://zenodo.org/badge/110861368.svg)](https://zenodo.org/badge/latestdoi/110861368)

# Materials Cloud - Empa Surfaces App

This [Materials Cloud jupyter](https://jupyter.materialscloud.org) app is a GUI for
AiiDA workflows that allows to easily compute and plot in a standardized way
basic DFT properties for graphene-based nanoribbons.

The nanoribbon workflow can be subdivided in three sections:
 * Upload/ modification of Structures
 * Calculation of properties
 * Search / visualization in the database

To get an idea of how it works, please [check out the videos](https://www.youtube.com/playlist?list=PL19kfLn4sO_8O_yQTL6KK0nC2adrrLqmi).

## Uploading a new structure
File formats compatible with [ASE](https://wiki.fysik.dtu.dk/ase/) can be used.
In the example a .mol file for a structure is generated with ChemDraw. 

The structure is uploaded, a warning message is provided if similar structures were already computed.

The structure is stored in the database.

The structure created with ChemDraw is properly scaled to conventional C-C sp2 bond lengths.

A unit cell is created identifying two equivalent atoms in the edited (scaled) structure.

## Submission of a calculation

 1. Select from the list of edited structure a structure that has a unit cell defined (it is also possible to directly upload structures with predefined unit cell thus skipping the steps 3-4 of the upload procedure)
 1. Select a computer from the list of installed computers
 1. Select the desired precision in terms of k-point samplig (1 corresponds to 60 k-pt for a unit cell of size 2.4 A)
 1. Submit the calculation

The status of the workflow can be monitored either from the AiiDA status or from the list of submitted workflows.

## Contact

For inquiries concerning Materials Cloud, please use the [public issue tracker](https://github.com/materialscloud-org/issues).
For inquiries concerning the Empa Surfaces App, please contact [carlo.pignedoli@empa.ch](mailto:carlo.pignedoli@empa.ch).
