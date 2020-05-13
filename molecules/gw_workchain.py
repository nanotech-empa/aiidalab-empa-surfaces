from aiida.engine import WorkChain, ToContext
from aiida_cp2k.calculations import Cp2kCalculation
from apps.surfaces.widgets.create_xyz_input_files import make_geom_file
from aiida.orm import SinglefileData
from aiida.orm import Code

class GwWorkChain(WorkChain):
    """Compute Band Structure of a material."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('xyz_scf', valid_type=SinglefileData)
        spec.input('xyz_gw', valid_type=SinglefileData)
        spec.input('code', valid_type=Code)
        spec.input('parameters_scf', valid_type=Dict)
        spec.input('parameters_gw', valid_type=Dict)
        spec.outline(
            cls.run_scf,
            cls.run_gw,
            cls.return_results
        )

        
        
    def run_scf(self):
        
        #calc1                
        builder = Cp2kCalculation.get_builder()
        # code 
        builder.cp2k.code = self.code

        # create input xyz structure
        builder.cp2k.file.input_xyz = self.inputs.xyz_scf        
        builder.metadata = self.inputs.parameters_scf['metadata']
        builder.metadata.label = self.inputs.parameters_scf['workchain']
        builder.metadata.description = self.inputs.parameters_scf['description']
        builder.metadata['label'] = 'scf_calculation'
        builder.metadata['description'] = self.inputs.parameters_scf['description']
        builder.parameters = Dict(dict=submit_dict)
        # Create the calculation process and launch it
        running = self.submit(builder)
        self.report("Submitted Cp2k calculation for SCF")
        return ToContext(calculation=running)
    
    def run_gw(self):        
        #calc2
        #new structure with maybe ghost atoms
        restart_folder = self.ctx.calculation.outputs.remote_folder
        builder = Cp2kCalculation.get_builder()
        # code 
        builder.cp2k.code = self.code
        # create input xyz structure
        builder.cp2k.file.input_xyz = self.inputs.xyz_gw         
        builder.parent_calc_folder = restart_folder
        builder.metadata = self.inputs.parameters_scf['metadata']
        builder.metadata.label = self.inputs.parameters_scf['workchain']
        builder.metadata.description = self.inputs.parameters_scf['description']
        builder.metadata['label'] = 'gw_calculation'
        builder.metadata['description'] = self.inputs.parameters_scf['description']
        builder.parameters = Dict(dict=submit_dict)
        # Create the calculation process and launch it
        running = self.submit(builder)
        self.report("Submitted Cp2k calculation for GW")
        return ToContext(calculation=running)        
        
    def return_results(self):
        ##packing resutks and returning
        return