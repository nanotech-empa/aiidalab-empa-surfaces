from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.base import Int, Float, Str
from aiida.orm.data.structure import StructureData
from aiida.orm.data.upf import get_pseudos_dict, get_pseudos_from_structure

from aiida.orm.calculation.job.quantumespresso.pw import PwCalculation
from aiida.orm.calculation.job.quantumespresso.pp import PpCalculation
from aiida.orm.calculation.job.quantumespresso.projwfc import ProjwfcCalculation

from aiida.orm import load_node
from aiida.orm.code import Code
from aiida.orm.computer import Computer
from aiida.orm.querybuilder import QueryBuilder

from aiida.work.workchain import WorkChain, ToContext, Calc
from aiida.work.run import run, async, submit

import numpy as np

class NanoribbonWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(NanoribbonWorkChain, cls).define(spec)
        spec.input("pw_code", valid_type=Code)
        spec.input("pp_code", valid_type=Code)
        spec.input("projwfc_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("precision", valid_type=Float, default=Float(1.0), required=False)
        spec.outline(
            cls.run_cell_opt1,
            cls.run_cell_opt2,
            cls.run_scf,
            #cls.run_export_pdos,
            cls.run_export_hartree,
            cls.run_bands,
            cls.run_export_pdos,
            cls.run_bands_lowres,
            cls.run_export_orbitals,
            cls.run_export_spinden,
        )
        spec.dynamic_output()

    #============================================================================================================
    def run_cell_opt1(self):
        structure = self.inputs.structure
        return self._submit_pw_calc(structure, label="cell_opt1", runtype='vc-relax', precision=0.5, min_kpoints=1)

    #============================================================================================================
    def run_cell_opt2(self):
        prev_calc = self.ctx.cell_opt1
        self._check_prev_calc(prev_calc)
        structure = prev_calc.out.output_structure
        return self._submit_pw_calc(structure, label="cell_opt2", runtype='vc-relax', precision=1.0, min_kpoints=1)

    #============================================================================================================
    def run_scf(self):
        prev_calc = self.ctx.cell_opt2
        self._check_prev_calc(prev_calc)
        structure = prev_calc.out.output_structure
        return self._submit_pw_calc(structure, label="scf", runtype='scf', precision=3.0, min_kpoints=10, wallhours=4)


    #============================================================================================================
    def run_export_hartree(self):
        self.report("Running pp.x to export hartree potential")

        inputs = {}
        inputs['_label'] = "export_hartree"
        inputs['code'] = self.inputs.pp_code

        prev_calc = self.ctx.scf
        self._check_prev_calc(prev_calc)
        inputs['parent_folder'] = prev_calc.out.remote_folder

        structure = prev_calc.inp.structure
        cell_a = structure.cell[0][0]
        cell_b = structure.cell[1][1]
        cell_c = structure.cell[2][2]

        parameters = ParameterData(dict={
                  'inputpp':{
                      'plot_num': 11, # the V_bare + V_H potential
                  },
                  'plot':{
                      'iflag': 2, # 2D plot
                      'output_format': 7, # format suitable for gnuplot   (2D) x, y, f(x,y)
                      'x0(1)': 0.0, #3D vector, origin of the plane (in alat units)
                      'x0(2)': 0.0,
                      'x0(3)': cell_c/cell_a,
                      'e1(1)': cell_a/cell_a, #3D vectors which determine the plotting plane (in alat units)
                      'e1(2)': 0.0,
                      'e1(3)': 0.0,
                      'e2(1)': 0.0,
                      'e2(2)': cell_b/cell_a,
                      'e2(3)': 0.0,
                      'nx': 10, # Number of points in the plane
                      'ny': 10,
                      'fileout': 'vacuum_hartree.dat',
                  },
        })
        inputs['parameters'] = parameters

        settings = ParameterData(dict={'additional_retrieve_list':['vacuum_hartree.dat']})
        inputs['settings'] = settings

        inputs['_options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 20 * 60,
            # workaround for flaw in PpCalculator. We don't want to retrive this huge intermediate file.
            "append_text": u"rm -v aiida.filplot\n",
        }

        future = submit(PpCalculation.process(), **inputs)
        return ToContext(hartree=Calc(future))


    #============================================================================================================
    def run_bands(self):
        prev_calc = self.ctx.scf
        self._check_prev_calc(prev_calc)
        structure = prev_calc.inp.structure
        parent_folder = prev_calc.out.remote_folder
        return self._submit_pw_calc(structure, label="bands", parent_folder=parent_folder, runtype='bands', 
                                    precision=4.0, min_kpoints=20, wallhours=6)


    #============================================================================================================
    def run_bands_lowres(self):
        prev_calc = self.ctx.scf
        self._check_prev_calc(prev_calc)
        structure = prev_calc.inp.structure
        parent_folder = prev_calc.out.remote_folder
        return self._submit_pw_calc(structure, label="bands_lowres", parent_folder=parent_folder, runtype='bands', 
                                    precision=0.0, min_kpoints=12, wallhours=2)


    #============================================================================================================
    def run_export_orbitals(self):
        self.report("Running pp.x to export KS orbitals")

        inputs = {}
        inputs['_label'] = "export_orbitals"
        inputs['code'] = self.inputs.pp_code
        prev_calc = self.ctx.bands_lowres
        self._check_prev_calc(prev_calc)
        inputs['parent_folder'] = prev_calc.out.remote_folder

        nel = prev_calc.res.number_of_electrons
        nkpt = prev_calc.res.number_of_k_points
        nbnd = prev_calc.res.number_of_bands
        nspin = prev_calc.res.number_of_spin_components
        volume = prev_calc.res.volume
        kband1 = max(int(nel/2) - 6,1)
        kband2 = min(int(nel/2) + 7,nbnd)
        kpoint1 = 1
        kpoint2 = nkpt * nspin
        nhours = 2 + min(22,2*int(volume/1500))

        parameters = ParameterData(dict={
                  'inputpp':{
                      'plot_num': 7, # contribution of a selected wavefunction to charge density
                      'kpoint(1)': kpoint1,
                      'kpoint(2)': kpoint2,
                      'kband(1)': kband1,
                      'kband(2)': kband2,
                  },
                  'plot':{
                      'iflag': 3, # 3D plot
                      'output_format': 6, # CUBE format
                      'fileout': '_orbital.cube',
                  },
        })
        inputs['parameters'] = parameters

        inputs['_options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": nhours* 60 *60, # 6 hours
            "append_text": self._get_cube_cutter(),
        }

        settings = ParameterData(dict={'additional_retrieve_list':['*.cube.gz']})
        inputs['settings'] = settings

        future = submit(PpCalculation.process(), **inputs)
        return ToContext(orbitals=Calc(future))


    #============================================================================================================
    def run_export_spinden(self):
        self.report("Running pp.x to compute spinden")

        inputs = {}
        inputs['_label'] = "export_spinden"
        inputs['code'] = self.inputs.pp_code
        prev_calc = self.ctx.scf
        self._check_prev_calc(prev_calc)
        inputs['parent_folder'] = prev_calc.out.remote_folder

        nspin = prev_calc.res.number_of_spin_components
        if nspin == 1:
            self.report("Skipping, got only one spin channel")
            return

        parameters = ParameterData(dict={
                  'inputpp':{
                      'plot_num': 6, # spin polarization (rho(up)-rho(down))

                  },
                  'plot':{
                      'iflag': 3, # 3D plot
                      'output_format': 6, # CUBE format
                      'fileout': '_spin.cube',
                  },
        })
        inputs['parameters'] = parameters

        inputs['_options'] = {
            "resources": {"num_machines": 1},
            "max_wallclock_seconds": 30 * 60, # 30 minutes
            "append_text": self._get_cube_cutter(),
        }

        settings = ParameterData(dict={'additional_retrieve_list':['*.cube.gz']})
        inputs['settings'] = settings

        future = submit(PpCalculation.process(), **inputs)
        return ToContext(spinden=Calc(future))


    #============================================================================================================
    def run_export_pdos(self):
        self.report("Running projwfc.x to export PDOS")
        
        inputs = {}
        inputs['_label'] = "export_pdos"
        inputs['code'] = self.inputs.projwfc_code
        prev_calc = self.ctx.bands
        self._check_prev_calc(prev_calc)
        volume = prev_calc.res.volume
        nnodes = max(1, int(volume/1500))
        nhours = 2 + min(22,2*int(volume/1500))
        inputs['parent_folder'] = prev_calc.out.remote_folder
        
        parameters = ParameterData(dict={
                  'projwfc':{
                      'ngauss': 1,
                      'degauss': 0.007,
                      'DeltaE': 0.01,
                      'filproj': 'projection.out',
                      #'filpdos' : 'totdos',
                      #'kresolveddos': True,
                  },
        })
        inputs['parameters'] = parameters

        inputs['_options'] = {
            "resources": {"num_machines": nnodes, "num_mpiprocs_per_machine": 1},
            "max_wallclock_seconds":  nhours * 60 * 60, # 12 hours
        }

        settings = ParameterData(dict={'additional_retrieve_list':['./out/aiida.save/atomic_proj.xml', '*_up', '*_down', '*_tot']})
        inputs['settings'] = settings

        future = submit(ProjwfcCalculation.process(), **inputs)
        return ToContext(pdos=Calc(future))


    #============================================================================================================
    def _check_prev_calc(self, prev_calc):
        error = None
        if prev_calc.get_state() != 'FINISHED':
            error = "Previous calculation in state: "+prev_calc.get_state()
        elif "aiida.out" not in prev_calc.out.retrieved.get_folder_list():
            error = "Previous calculation did not retrive aiida.out"
        else:
            fn = prev_calc.out.retrieved.get_abs_path("aiida.out")
            content = open(fn).read()
            if "JOB DONE." not in content:
                error = "Previous calculation's aiida.out does not contain JOB DONE."
        if error:
            self.report("ERROR: "+error)
            self.abort(msg=error)
            raise Exception(error)


    #============================================================================================================
    def _submit_pw_calc(self, structure, label, runtype, precision, min_kpoints, wallhours=24, parent_folder=None):
        self.report("Running pw.x for "+label)

        inputs = {}
        inputs['_label'] = label
        inputs['code'] = self.inputs.pw_code
        inputs['structure'] = structure
        inputs['parameters'] =  self._get_parameters(structure, runtype)
        inputs['pseudo'] = self._get_pseudos(structure, family_name="SSSP_acc_PBE")
        if parent_folder:
            inputs['parent_folder'] = parent_folder

        # kpoints
        cell_a = inputs['structure'].cell[0][0]
        precision *= self.inputs.precision.value
        nkpoints = max(min_kpoints, int(30 * 2.5/cell_a * precision))
        use_symmetry = runtype != "bands"
        kpoints = self._get_kpoints(nkpoints, use_symmetry=use_symmetry)
        inputs['kpoints'] = kpoints

        # parallelization settings
        npools = min(1+nkpoints/5, 5)
        natoms = len(structure.sites)
        nnodes = (1 + natoms/60) * npools
        inputs['_options'] = {
            "resources": {"num_machines": nnodes},
            "max_wallclock_seconds": wallhours * 60 * 60, # hours
        }
        settings = {'cmdline': ["-npools",str(npools)]}

        if runtype == "bands":
            settings['also_bands'] = True # instruction for output parser

        inputs['settings'] = ParameterData(dict=settings)

#         self.report("precision %f"%precision)
#         self.report("nkpoints %d"%nkpoints)
#         self.report("npools %d"%npools)
#         self.report("natoms %d"%natoms)
#         self.report("nnodes %d"%nnodes)

        future = submit(PwCalculation.process(), **inputs)
        return ToContext(**{label:Calc(future)})

    #============================================================================================================
    def _get_parameters(self, structure, runtype):
        params = {'CONTROL': {
                     'calculation': runtype,
                     'wf_collect': True,
                     'forc_conv_thr': 0.0001,
                     'nstep': 500,
                     },
                 'SYSTEM': {
                     'ecutwfc': 50.,
                     'ecutrho': 400.,
                     'occupations': 'smearing',
                     'degauss': 0.001,
                     },
                 'ELECTRONS': {
                     'conv_thr': 1.e-8,
                     'mixing_beta': 0.25,
                     'electron_maxstep': 50,
                     'scf_must_converge': False,
                     },
        }

        if runtype == "vc-relax":
            params['CELL'] = {'cell_dofree': 'x'} # in y and z direction there is only vacuum

        #if runtype == "bands":
        #    params['CONTROL']['restart_mode'] = 'restart'

        start_mag = self._get_magnetization(structure)
        if any([m!=0 for m in start_mag.values()]):
            params['SYSTEM']['nspin'] = 2
            params['SYSTEM']['starting_magnetization'] = start_mag

        return ParameterData(dict=params)

    #============================================================================================================
    def _get_kpoints(self, nx, use_symmetry=True):
        nx = max(1, nx)
        
        kpoints = KpointsData()
        if use_symmetry:
            kpoints.set_kpoints_mesh([nx, 1, 1], offset=[0.0, 0.0, 0.0] )
        else:
            # list kpoints explicitly
            points = [[r, 0.0, 0.0] for r in np.linspace(0, 0.5, nx)]
            kpoints.set_kpoints(points)
            
        return kpoints

    #============================================================================================================
    def _get_pseudos(self, structure, family_name):
        kind_pseudo_dict = get_pseudos_from_structure(structure, family_name)
        pseudos = {}
        for p in kind_pseudo_dict.values():
            kinds = "_".join([k for k, v in kind_pseudo_dict.items() if v==p])
            pseudos[kinds] = p

        return pseudos

    #============================================================================================================
    def _get_magnetization(self, structure):
        start_mag={}
        for i in structure.kinds:
            if i.name.endswith("1"):
                start_mag[i.name] = 1.0
            elif i.name.endswith("2"):
                start_mag[i.name] = -1.0
            else:
                start_mag[i.name] = 0.0
        return start_mag

    #============================================================================================================
    def _get_cube_cutter(self):
        append_text = ur"""
cat > postprocess.py << EOF

from glob import glob
import numpy as np
import gzip

for fn in glob("*.cube"):
    # parse
    lines = open(fn).readlines()
    header = np.fromstring("".join(lines[2:6]), sep=' ').reshape(4,4)
    natoms, nx, ny, nz = header[:,0].astype(int)
    cube = np.fromstring("".join(lines[natoms+6:]), sep=' ').reshape(nx, ny, nz)

    # plan
    dz = header[3,3]
    angstrom = int(1.88972 / dz)
    z0 = nz/2 + 1*angstrom # start one angstrom above surface
    z1 = z0   + 3*angstrom # take three layers at one angstrom distance
    zcuts = range(z0, z1+1, angstrom)

    # output
    lines[2] = "%5.d 0.0 0.0 %f\n"%(natoms,  z0*dz) # change offset header
    lines[5] = "%6.d 0.0 0.0 %f\n"%(len(zcuts), angstrom*dz) # change shape header
    with gzip.open(fn+".gz", "w") as f:
        f.write("".join(lines[:natoms+6])) # write header
        np.savetxt(f, cube[:,:,zcuts].reshape(-1, len(zcuts)), fmt="%.5e")
EOF

python ./postprocess.py
"""
        return append_text

#EOF