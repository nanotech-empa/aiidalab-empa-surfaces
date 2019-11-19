from aiida.orm import CalcFunctionNode, CalcJobNode, Node, QueryBuilder, WorkChainNode, StructureData
from apps.surfaces.widgets import  thumbnails,find_mol

def preprocess_newbies(preprocess_v=0.0,wlabel='',clabel=''):
    qb = QueryBuilder()
    qb.append(WorkChainNode, filters={
        'label': wlabel,
        'or':[
               {'extras': {'!has_key': 'preprocess_version'}},
               {'extras.preprocess_version': {'<': preprocess_v}},
           ],
    })
    
    for m in qb.all(): # iterall() would interfere with set_extra()
        n = m[0]
        if not n.is_sealed:
            print("Skipping underway workchain PK %d"%n.pk)
            continue
        if 'obsolete' not in n.extras:
            n.set_extra('obsolete', False)
        try:
            preprocess_one(n,label=clabel)
            n.set_extra('preprocess_successful', True)
            n.set_extra('preprocess_error', '')
            n.set_extra('preprocess_version', preprocess_v)
            print("Preprocessed PK %d"%n.pk)
        except Exception as e:
            n.set_extra('preprocess_successful', False)
            n.set_extra('preprocess_error', str(e))
            n.set_extra('preprocess_version', preprocess_v)
            print("Failed to preprocess PK %d: %s"%(n.pk, e))
            
            
def preprocess_one(workcalc,label=''):
   
    def get_calc_by_label(workcalc, label):
        qb = QueryBuilder()
        qb.append(WorkChainNode, filters={'uuid':workcalc.uuid})
        qb.append(CalcJobNode, with_incoming=WorkChainNode, filters={'label':label})
        #qb.order_by({'calc':[{'id':{'order':'desc'}}]})
        if qb.count() == 0:
            raise(Exception("Could not find %s calculation."%label))
        calc = qb.all()[0][0]
        return calc

    # formula
    structure = workcalc.inputs.cp2k__file__input_xyz.get_incoming().all_nodes()[0].get_incoming().all_nodes()[1]
    ase_struct = structure.get_ase()
    
    res = find_mol.analyze_slab(ase_struct)
    mol_formula=''
    for imol in res['all_molecules']:
        mol_formula+=ase_struct[imol].get_chemical_formula()+' '
    if len(res['slabatoms'])>0:
        slab_formula=ase_struct[res['slabatoms']].get_chemical_formula()
        if len(res['bottom_H']) >0:
            slab_formula+=' saturated: ' + ase_struct[res['bottom_H']].get_chemical_formula()
        if len(res['adatoms']) >0:
            slab_formula+=' adatoms: ' + ase_struct[res['adatoms']].get_chemical_formula()  
        workcalc.set_extra('formula', '{} at {}'.format(mol_formula,slab_formula))
    else:
        formula = ase_struct.get_chemical_formula()
        workcalc.set_extra('formula', '{}'.format(formula))
    
    workcalc.set_extra('structure_description', structure.description)    
    
    
    # optimized structure
    calc = get_calc_by_label(workcalc, label) # TODO deal with restarts, check final state
    opt_structure = calc.outputs.output_structure
    workcalc.set_extra('opt_structure_uuid', calc.outputs.output_structure.uuid)
    workcalc.set_extra('energy', calc.res.energy)

    # thumbnail
    thumbnail = thumbnails.render_thumbnail(ase_struct)
    workcalc.set_extra('thumbnail', thumbnail)