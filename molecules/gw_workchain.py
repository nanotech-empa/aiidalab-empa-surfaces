from aiida.engine import WorkChain, ToContext
from aiida_cp2k.calculations import Cp2kCalculation
from apps.surfaces.widgets.create_xyz_input_files import make_geom_file
from aiida.orm import SinglefileData, Dict
from aiida.orm import Code
import numpy as np
from copy import deepcopy

class GwWorkChain(WorkChain):
    """Compute GW for a molecule."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('xyz_gw', valid_type=SinglefileData)
        spec.input('code', valid_type=Code)
        spec.input('final_dict', valid_type=Dict)
        spec.input('submit_dict', valid_type=Dict)
        spec.outline(
            cls.run_scf,
            cls.validate_results,
            cls.run_gw,
            cls.validate_results,
            cls.return_results
        )
        spec.exit_code(300, 'CALC_FAILED', message='The calculation failed.')
        
    #calc1    
    def run_scf(self):
        
        
        submit_dict_scf = deepcopy(self.inputs.submit_dict.get_dict())
        
        ## create from the GW inp a simplified input for SCF calc
        submit_dict_scf['FORCE_EVAL']['DFT']['XC'] = {'XC_FUNCTIONAL': {'_': 'PBE'} }
        del(submit_dict_scf['FORCE_EVAL']['DFT']['POISSON'])
        del(submit_dict_scf['FORCE_EVAL']['SUBSYS']['CELL']['PERIODIC']) 
        #submit_dict_scf['FORCE_EVAL']['DFT']['QS']={'METHOD': 'GAPW'}
        #cell GW is *2 +15 of molecule size
        cella = np.fromstring(submit_dict_scf['FORCE_EVAL']['SUBSYS']['CELL']['A'],dtype=float,sep=' ')/2+3.5 
        cellb = np.fromstring(submit_dict_scf['FORCE_EVAL']['SUBSYS']['CELL']['B'],dtype=float,sep=' ')/2+3.5
        cellc = np.fromstring(submit_dict_scf['FORCE_EVAL']['SUBSYS']['CELL']['C'],dtype=float,sep=' ')/2+3.5
        del submit_dict_scf['FORCE_EVAL']['SUBSYS']['CELL']['A']
        del submit_dict_scf['FORCE_EVAL']['SUBSYS']['CELL']['B']
        del submit_dict_scf['FORCE_EVAL']['SUBSYS']['CELL']['C']
        cell = str(cella[0]) +' '+str(cellb[1]) +' '+str(cellc[2]) 
        submit_dict_scf['FORCE_EVAL']['SUBSYS']['CELL']['ABC'] = cell   
        submit_dict_scf['FORCE_EVAL']['DFT']['SCF']['LEVEL_SHIFT'] = 0.1
        del(submit_dict_scf['FORCE_EVAL']['DFT']['SCF']['EPS_EIGVAL'])
        del(submit_dict_scf['FORCE_EVAL']['DFT']['SCF']['OT'])
        del(submit_dict_scf['FORCE_EVAL']['DFT']['SCF']['OUTER_SCF'])
        
                        
        builder = Cp2kCalculation.get_builder()
        # code 
        builder.code = self.inputs.code
        builder.parameters = Dict(dict=submit_dict_scf)
        
        builder.file = {
            'basis'     : SinglefileData(file='/home/aiida/apps/surfaces/Files/GW_BASIS_SET'),
            'pseudo'    :  SinglefileData(file='/home/aiida/apps/surfaces/Files/ALL_POTENTIALS'),
            'input_xyz' : self.inputs.xyz_gw
        }
        builder.metadata.options.resources = {
            "num_machines": self.inputs.final_dict['num_machines_scf'],
            "num_mpiprocs_per_machine": self.inputs.final_dict['num_mpiprocs_per_machine_scf'],
        }
        builder.metadata.options.max_wallclock_seconds = 5*3600        
        builder.metadata.options['withmpi'] = True
        builder.metadata.label = 'scf_step'
        builder.metadata.description = self.inputs.final_dict['description']      
        
        # Create the calculation process and launch it
        running = self.submit(builder)
        self.report("Submitted Cp2k calculation for SCF")
        return ToContext(calculation=running)
    
#CHECK VALIDITY....
#content_string = calc.outputs.retrieved.get_object_content(calc.get_attribute('output_filename'))    
    
    def run_gw(self):        
        #calc2
               
        restart_folder = self.ctx.calculation.outputs.remote_folder
        builder = Cp2kCalculation.get_builder()
        # code 
        builder.code = self.inputs.code
        submit_dict = deepcopy(self.inputs.submit_dict.get_dict())
        builder.parameters = Dict(dict=submit_dict)
        
        builder.file = {
            'basis'     :  SinglefileData(file='/home/aiida/apps/surfaces/Files/GW_BASIS_SET'),
            'pseudo'    :  SinglefileData(file='/home/aiida/apps/surfaces/Files/ALL_POTENTIALS'),
            'input_xyz' : self.inputs.xyz_gw
        }
        builder.metadata.options.resources = {
            'num_machines' : self.inputs.final_dict['metadata']['options']['resources']['num_machines'],
            'num_mpiprocs_per_machine' : self.inputs.final_dict['metadata']['options']['resources']['num_mpiprocs_per_machine'],
            'num_cores_per_mpiproc' : self.inputs.final_dict['metadata']['options']['resources']['num_cores_per_mpiproc']
        }
        builder.metadata.options['withmpi'] = True
        builder.metadata.options.max_wallclock_seconds = self.inputs.final_dict['metadata']['options']['max_wallclock_seconds']      
        
        builder.metadata.label = 'gw_step'
        builder.metadata.description = self.inputs.final_dict['description'] 
        builder.parent_calc_folder = restart_folder

        # Create the calculation process and launch it
        running = self.submit(builder)
        self.report("Submitted Cp2k calculation for GW")
        return ToContext(calculation=running)        
    
    def validate_results(self):  
        if self.ctx.calculation.exit_status != 0:
            return self.exit_codes.CALC_FAILED        
        
    def return_results(self):
        ##packing results and returning
        return