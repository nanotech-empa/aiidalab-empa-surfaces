# Export simulated STM maps to openBIS microscopy

Open a completed calculation in **View STM**, select the stored series and
energies, then choose **Prepare microscopy export**. The shared openBIS review
opens below the plots. Select the project/experiment, review the simulation
and its links, then press **Export to openBIS**. Preparation alone writes nothing.
You need the existing openBIS connection configured in the ELN app.

The existing `SPM_SIMULATION` object keeps the calculation and AiiDA archive.
Microscopy maps are linked `IMAGING_DATA` datasets with Nanonis-compatible SXM
files, so they use the established microscopy viewer. No new object types or
schema properties are required. Each map includes its original selected array
and grid in a portable NPZ and a JSON provenance sidecar.

- **STM at constant height:** the channel is simulated current in native
  arbitrary units. The requested height, energy relative to the Fermi level,
  broadening and tip parameters are retained. There is no inferred current in A.
- **STM at constant current** (constant isovalue in the simulation): the channel is
  height above the highest atom, with the native isovalue retained. SXM stores
  heights in metres; the original map remains in Å. This is a unit conversion,
  not a current calibration. **STS maps represent dI/dV**, at constant height
  or on the constant-current surface, in native arbitrary units.
- **Oblique/rotated grids:** a derived Cartesian raster uses inverse-grid linear
  interpolation, like the shared cube slicer. Points outside the original
  domain are NaN. The original grid/data are retained without resampling.
- **Optional curves:** explicitly select *Include image-area mean curves* and
  at least two energies for each series. DAT contains energy versus the mean of
  finite samples of the original map, with the native quantity and unit.

Plot colors, symmetry and display normalization do not modify exported values.
Export one calculation at a time; plotting several collections remains supported.
A repeated export checks dataset identity in openBIS and reuses the existing
map, including when local AiiDA extras were reset. After an interrupted preview,
use **Retry incomplete steps**; do not create another simulation to repair it.
Concurrent exports of the same selection from independent clients should be
avoided: the server does not enforce uniqueness of this metadata key.

SXM compatibility requires two direction buffers in the established parser.
The second buffer is a reversed encoding of the *same* map, not a second scan
or a separately computed simulation. Header and JSON explicitly record this.
Bias and the original simulation setpoint (3D-field isovalue or height) are
explicit. Instrument timing/gain placeholders are marked not applicable.

## From the AI assistant

Ask, for example: “Export the −0.5 and +0.5 eV constant-height maps to openBIS
microscopy.” The assistant reads the visible series/energy controls, prepares
this same review with `app_prepare_export`, and asks for confirmation of the
current destination and action. It never needs your token in chat. A changed
selection/destination requires a fresh inspection/review. Verify the returned
microscopy dataset IDs and preview status; a button callback alone is not proof
of publication.

## Qualification

Synthetic round-trips cover array orientation, units, Cartesian/oblique grids,
original-data retention, and retry identity. The microscopy server still needs
its installed Nanonis adaptors and DSS dataset service. In particular, older
server parser bundles may label an unknown simulated channel `N/A`; do not
reinterpret it as amperes. Confirm the actual server viewer/preview before
calling that deployment qualified.
