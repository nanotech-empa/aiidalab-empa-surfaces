# Export simulated STM maps to openBIS microscopy

Open a completed calculation in **View STM**, select the stored series and
energies, then choose **Prepare microscopy export**. The shared openBIS review
opens below the plots. Select the project/experiment, review the simulation
and its links, then press **Export to openBIS**. Preparation alone writes nothing.
You need the existing openBIS connection configured in the ELN app.

Each distinct selected series/bias pair is one microscopy map dataset. For
example, one constant-current series at −3 V gives one dataset; two series at
two biases give four. The review counts maps and any explicitly drawn local
spectra/profile groups and curves. It freezes the selection at **Prepare microscopy export**; prepare
again after changing series/energies. Previously plotted images do not add
more selections. Existing matching server datasets are reused.

The comparison gallery intentionally scrolls **horizontally**: bias columns,
selected cases on rows within each column. Its height follows the cases; there
is no vertical scrollbar inside the gallery.

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
- **SXM area:** set width, height and XY centre in Å. **Use simulation cell**
  restores the smallest Cartesian rectangle containing the simulation cell's
  A/B parallelogram, anchored to the simulation coordinates (not the molecule).
  For oblique cells, its corner regions are filled from periodic neighbouring
  cells. The native CP2K `CELL.PERIODIC` setting determines which lattice
  directions can wrap; source-structure PBC flags alone are not used.
  Linear inverse-grid interpolation samples Cartesian pixel centres, including
  across periodic seams. Nonperiodic regions outside the computed map stay NaN.
  A cropped map covering less than a whole required periodic domain cannot
  supply that missing data and is rejected. Original arrays and full grid
  metadata remain intact. Area changes require a new review and identify a
  different SXM dataset; they do not move atoms or recenter the simulation.

## STS point spectra and STM line profiles

Open **Extract STS point spectra or STM line profiles (optional)**. Choose the
stored series under **Analyse**, then **Open sampling map**. Its rectangular
view uses the same physical XY coordinates and native PBC as SXM export.

- **STS series:** click one or more points. P1, P2, … have different colours
  matching their dI/dV-versus-bias curves. Each spectrum uses all stored energies
  at that fixed XY point, without averaging the image or interpolating energies.
  Changing **Map bias [V]** changes the reference image; it does not truncate
  the spectrum. A constant-current STS series remains dI/dV sampled on its
  stored constant-isovalue surface.
- **STM series:** click a start and an end for each L1, L2, … line. A
  constant-current (constant-isovalue) map gives **height versus distance**;
  a constant-height map gives **simulated current versus distance**. The profile
  is for the selected bias. Changing bias updates it along the same physical
  line. Both endpoints are included; linear sampling intervals are no larger
  than one native grid step. Extra interpolated samples do not add resolution.
- Select a marker and **Remove selected marker**, or **Clear points/lines**.
  The exact-coordinate section accepts Å coordinates instead of mouse clicks:
  add one coordinate per point, or two in sequence for a line. Changing the
  source/analysis series clears the old selections. Incomplete or invalid
  lines must be completed/removed before preparing an export.
- **Download DAT files** creates a ZIP of the curves and their sampling
  coordinates/provenance, without publishing. It contains one file per point
  or line. Spectra have bias [V] and native simulated dI/dV; profiles have
  distance [Å] and height [Å] or native simulated current. Colours, labels,
  source identity, grid/PBC, line endpoints and selected bias are recorded.

Drawn curves are included when **Prepare microscopy export** is pressed.
The review counts SXM datasets, grouped DAT datasets and individual curves.
One STS series is one DAT dataset containing its point files; one STM series
at one bias is one DAT dataset containing its profile files. Microscopy's
existing grouping/legend controls select those files. Its colormap remains
under microscopy-viewer control; selected marker colours are retained in the
metadata. Changing markers invalidates a prior review. Clear markers for an
images-only export. SXM crop/size changes preserve physical sampling coordinates.
There is no global-image-average checkbox.


Plot colors, symmetry and display normalization do not modify exported values.
Export one calculation at a time; plotting several collections remains supported.
A repeated export checks dataset identity in openBIS and reuses the existing
simulation, including when local AiiDA extras were reset. The first export
includes a finished geometry optimization when it produced the stored STM
input. For an already exported STM simulation, **Add DAT to openBIS** shows the
destination and adds only new point spectra or line profiles to that object.
Use **Retry incomplete steps** to finish an interrupted first export.
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
current destination, SXM area/native PBC, dataset counts and action. It never needs your token in chat. A changed
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
