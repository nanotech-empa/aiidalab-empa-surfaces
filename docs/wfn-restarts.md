# CP2K wavefunction restart lookup

Leave **WFN restart folder** empty to search automatically for a compatible WFN
in the structure's calculation history. Keep **Skip automatic WFN restart
search** unchecked when you want this reuse. The app searches without an
additional action and uses an available compatible WFN as the SCF starting guess.

Imported structures can refer to another account's scratch directory or an old,
unconfigured AiiDA computer. The app checks the candidate file using the computer
and account selected for the new calculation. It does not change the imported
computer, its authentication, or the original AiiDA data. If the file is missing
or unreadable, the app explains that it cannot reuse it and proceeds with a fresh
SCF. A connection/configuration failure on the selected destination remains an
error and needs attention.

A folder entered explicitly takes precedence over automatic discovery. It is
not silently replaced by a fresh-SCF choice if it cannot be used. The explicit
skip option still requests an independent fresh SCF.

This policy applies to SCF, geometry optimization, PDOS, SPM, phonons, adsorption
and replica-chain forms. NEB performs the equivalent search in its backend and
runs its initial SCF if none of the endpoint WFNs can be reused. Existing charge,
spin and supported-provenance compatibility checks still apply.
