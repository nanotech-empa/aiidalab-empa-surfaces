def input_is_valid(job_details={}):
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
    ###UKS CHECK
    total_charge = structure_details["total_charge"]
    if "charge" in details_dict.keys():
        total_charge += details_dict["charge"]
    if total_charge % 2:
        if "multiplicity" in details_dict.keys():
            if details_dict["multiplicity"] % 2:
                return (False, "odd charge and odd multiplicity")
            elif details_dict["multiplicity"] == 0:
                return (False, "odd charge and RKS")
        else:
            return (False, "odd charge and RKS")
    else:
        if "multiplicity" in details_dict.keys():
            if (
                not details_dict["multiplicity"] % 2
                and details_dict["multiplicity"] > 0
            ):
                return (False, "even charge and even multiplicity")

    if "calc_type" in details_dict.keys():
        if details_dict["calc_type"] in ["Mixed DFTB", "Mixed DFT"]:
            ## ADD CHECK CONTINUITY MOL INDEXES

            for el in structure_details["slab_elements"]:
                if el not in ["Ag", "Au", "Cu", "H"]:
                    return (False, "Wrong slab composition")

            if len(structure_details["adatoms"]) > 0:
                return (False, "Found Adatoms cannot do Mixed DFT")

            if len(structure_details["unclassified"]) > 0:
                return (False, "Found unclassified atoms")

            for the_mol in structure_details["all_molecules"]:
                if max(
                    [i for j in structure_details["all_molecules"] for i in j]
                ) > min(structure_details["slabatoms"] + structure_details["bottom_H"]):
                    return (False, "Molecule is not at the beginning of the structure.")

    return (True, "")
