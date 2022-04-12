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

## For maintainers

To create a new release, clone the repository, install development dependencies with `pip install '.[dev]'`, and then execute `bumpver update --major/--minor/--patch`.
This will:

  1. Create a tagged release with bumped version and push it to the repository.
  2. Trigger a GitHub actions workflow that creates a GitHub release.

Additional notes:

  - Use the `--dry` option to preview the release change.
  - The release tag (e.g. a/b/rc) is determined from the last release.
    Use the `--tag` option to switch the release tag.


## Contact

For inquiries concerning Materials Cloud, please use the [public issue tracker](https://github.com/materialscloud-org/issues).
For inquiries concerning the Empa Surfaces App, please contact [carlo.pignedoli@empa.ch](mailto:carlo.pignedoli@empa.ch).
