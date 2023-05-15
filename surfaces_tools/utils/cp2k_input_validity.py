def input_is_valid(job_details=None):
    odd_charge = job_details["slab_analyzed"]["total_charge"]
    rks = True
    if "charge" in job_details.keys():
        odd_charge += job_details["charge"]
        if "uks_switch" in job_details.keys():
            if job_details["uks_switch"] == "UKS":
                rks = False

    if odd_charge % 2 > 0 and rks:
        print("ODD CHARGE AND RKS")
        return False

    if job_details["workchain"] == "NEBWorkchain":
        if len(job_details["replica_pks"].split()) < 2:
            print("Please select at least two  replica_pks")
            return False

    return True


def validate_input(structure_details, details_dict):
    # UKS check.
    total_charge = structure_details["total_charge"]
    if "charge" in details_dict["dft_params"]:
        total_charge += details_dict["dft_params"]["charge"]
    if total_charge % 2:
        if "multiplicity" in details_dict["dft_params"]:
            if details_dict["dft_params"]["multiplicity"] % 2:
                return (False, "odd charge and odd multiplicity")
            elif details_dict["dft_params"]["multiplicity"] == 0:
                return (False, "odd charge and RKS")
        else:
            return (False, "odd charge and RKS")
    else:
        if "multiplicity" in details_dict["dft_params"]:
            if (
                not details_dict["dft_params"]["multiplicity"] % 2
                and details_dict["dft_params"]["multiplicity"] > 0
            ):
                return (False, "even charge and even multiplicity")

    return (True, "")
