def input_is_valid(job_details={}):
    odd_charge = job_details['slab_analyzed']['total_charge']
    rks = True
    if 'charge' in job_details.keys():
        odd_charge += job_details['charge']         
        if 'uks_switch' in job_details.keys():
            if  job_details['uks_switch'] == 'UKS':
                rks=False                       

    if  odd_charge%2 >0 and rks  :
        print("ODD CHARGE AND RKS")
        return False

    if job_details['workchain'] == 'NEBWorkchain':
        if len(job_details['replica_pks'].split()) < 2:
            print('Please select at least two  replica_pks')
            return False


    return True