import os

import numpy as np
from aiida import engine, orm


def get_calc_by_label(workcalc, label):
    qb = orm.QueryBuilder()
    qb.append(orm.WorkChainNode, filters={"uuid": workcalc.uuid})
    qb.append(engine.CalcJob, with_incoming=orm.WorkChainNode, filters={"label": label})
    assert qb.count() == 1
    calc = qb.first()[0]
    assert calc.is_finished_ok
    return calc


def get_slab_calc_info(struct_node):
    html = ""
    try:
        cp2k_calc = struct_node.creator
        opt_workchain = cp2k_calc.caller
        thumbnail = opt_workchain.extras["thumbnail"]
        description = opt_workchain.description
        struct_description = opt_workchain.extras["structure_description"]
        struct_pk = struct_node.pk

        html += "<style>#aiida_results td,th {padding: 5px}</style>"
        html += '<table border=1 id="geom_info" style="margin:0px;">'
        html += "<tr>"
        html += "<th> Structure description: </th>"
        html += f"<td> {struct_description} </td>"
        html += f'<td rowspan="2"><img width="100px" src="data:image/png;base64,{thumbnail}" title="PK:{struct_pk}"></td>'
        html += "</tr>"
        html += "<tr>"
        html += "<th> Calculation description: </th>"
        html += f"<td> {description} </td>"
        html += "</tr>"
        html += "</table>"

    except Exception:
        html = ""
    return html


def comp_plugin_codes(computer_name, plugin_name):
    qb = orm.QueryBuilder()
    qb.append(orm.Computer, project="name", tag="computer")
    qb.append(
        orm.Code,
        project="*",
        with_computer="computer",
        filters={
            "attributes.input_plugin": plugin_name,
            "or": [{"extras": {"!has_key": "hidden"}}, {"extras.hidden": False}],
        },
    )
    qb.order_by({orm.Code: {"id": "desc"}})
    codes = qb.all()
    sel_codes = []
    for code in codes:
        if code[0] == computer_name:
            sel_codes.append(code[1])
    return sel_codes


def create_stm_parameterdata(
    extrap_plane,
    const_height_text,
    struct_symbols,
    parent_dir,
    elim_float_slider0,
    elim_float_slider1,
    de_floattext,
    const_current_text,
    fwhms_text,
    ptip_floattext,
):
    max_height = max([float(h) for h in const_height_text])
    extrap_extent = max([max_height - extrap_plane, 5.0])

    # Evaluation region in z.
    z_min = "n-2.0_C" if "C" in struct_symbols else "p-4.0"
    z_max = f"p{extrap_plane:.1f}"

    energy_range_str = "{:.2f} {:.2f} {:.3f}".format(
        elim_float_slider0, elim_float_slider1, de_floattext
    )

    paramdata = {
        "--cp2k_input_file": parent_dir + "aiida.inp",
        "--basis_set_file": parent_dir + "BASIS_MOLOPT",
        "--xyz_file": parent_dir + "aiida.coords.xyz",
        "--wfn_file": parent_dir + "aiida-RESTART.wfn",
        "--hartree_file": parent_dir + "aiida-HART-v_hartree-1_0.cube",
        "--output_file": "stm.npz",
        "--eval_region": ["G", "G", "G", "G", z_min, z_max],
        "--dx": "0.15",
        "--eval_cutoff": "16.0",
        "--extrap_extent": str(extrap_extent),
        "--energy_range": energy_range_str.split(),
        "--heights": const_height_text,
        "--isovalues": const_current_text,
        "--fwhms": fwhms_text,
        "--p_tip_ratios": ptip_floattext,
    }
    return paramdata


def create_orbitals_parameterdata(
    extrap_plane,
    heights_text,
    parent_dir,
    n_homo_inttext,
    n_lumo_inttext,
    isovals_text,
    fwhms_text,
    ptip_floattext,
):
    max_height = max([float(h) for h in heights_text])
    extrap_extent = max([max_height - extrap_plane, 5.0])
    paramdata = {
        "--cp2k_input_file": parent_dir + "aiida.inp",
        "--basis_set_file": parent_dir + "BASIS_MOLOPT",
        "--xyz_file": parent_dir + "aiida.coords.xyz",
        "--wfn_file": parent_dir + "aiida-RESTART.wfn",
        "--hartree_file": parent_dir + "aiida-HART-v_hartree-1_0.cube",
        "--orb_output_file": "orb.npz",
        "--eval_region": ["G", "G", "G", "G", "n-1.0_C", "p%.1f" % extrap_plane],
        "--dx": "0.15",
        "--eval_cutoff": "16.0",
        "--extrap_extent": str(extrap_extent),
        "--n_homo": str(n_homo_inttext + 2),
        "--n_lumo": str(n_lumo_inttext + 2),
        "--orb_heights": heights_text,
        "--orb_isovalues": isovals_text,
        "--orb_fwhms": fwhms_text,
        "--p_tip_ratios": ptip_floattext,
    }
    return paramdata


def create_pp_parameterdata(
    ase_geom,
    dx,
    scanminz_floattxt,
    scanmaxz_floattxt,
    amp_floattxt,
    f0_cantilever_floattxt,
):
    cell = ase_geom.cell
    top_z = np.max(ase_geom.positions[:, 2])
    paramdata = {
        "probeType": "O",
        "charge": -0.028108681223969645,
        "sigma": 0.7,
        "tip": "s",
        "klat": 0.34901278868090491,
        "krad": 21.913190531846034,
        "r0Probe": [0.0, 0.0, 2.97],
        "PBC": "False",
        "gridA": list(cell[0]),
        "gridB": list(cell[1]),
        "gridC": list(cell[2]),
        "scanMin": [0.0, 0.0, np.round(top_z, 1) + scanminz_floattxt],
        "scanMax": [cell[0, 0], cell[1, 1], np.round(top_z, 1) + scanmaxz_floattxt],
        "scanStep": [dx, dx, dx],
        "Amplitude": amp_floattxt,
        "f0Cantilever": f0_cantilever_floattxt,
    }
    return paramdata


def create_2pp_parameterdata(
    ase_geom,
    dx,
    resp,
    scanminz_floattxt,
    scanmaxz_floattxt,
    amp_floattxt,
    f0_cantilever_floattxt,
):
    cell = ase_geom.cell
    top_z = np.max(ase_geom.positions[:, 2])
    paramdata = {
        "Catom": 6,
        "Oatom": 8,
        "ChargeCuUp": resp[0],
        "ChargeCuDown": resp[1],
        "Ccharge": resp[2],
        "Ocharge": resp[3],
        "sigma": 0.7,
        "Cklat": 0.24600212465950813,
        "Oklat": 0.15085476515590224,
        "Ckrad": 20,
        "Okrad": 20,
        "rC0": [0.0, 0.0, 1.82806112489999961213],
        "rO0": [0.0, 0.0, 1.14881347770000097341],
        "PBC": "False",
        "gridA": list(cell[0]),
        "gridB": list(cell[1]),
        "gridC": list(cell[2]),
        "scanMin": [0.0, 0.0, np.round(top_z, 1) + scanminz_floattxt],
        "scanMax": [cell[0, 0], cell[1, 1], np.round(top_z, 1) + scanmaxz_floattxt],
        "scanStep": [dx, dx, dx],
        "Amplitude": amp_floattxt,
        "f0Cantilever": f0_cantilever_floattxt,
        "tip": "None",
        "Omultipole": "s",
    }
    return paramdata


def create_hrstm_parameterdata(
    hrstm_code,
    parent_dir,
    ppm_dir,
    ase_geom,
    ppm_params_dict,
    tiptype_ipw,
    stip_ipw,
    pytip_ipw,
    pztip_ipw,
    pxtip_ipw,
    volmin_ipw,
    volmax_ipw,
    volstep_ipw,
    volstep_ipwmin,
    fwhm_ipw,
    wfnstep_ipw,
    extrap_ipw,
    workfun_ipw,
    orbstip_ipw,
    fwhmtip_ipw,
    rotate_ipw,
):
    # External folders.
    cell = orm.ArrayData()
    cell.set_array("cell", np.diag(ase_geom.cell))

    # PPM folder of position.
    ppm_qk = ppm_dir + "Qo{:1.2f}Qc{:1.2f}K{:1.2f}/".format(
        ppm_params_dict["Ocharge"], ppm_params_dict["Ccharge"], ppm_params_dict["Oklat"]
    )

    # Tip type to determine PDOS and PPM position files.
    if tiptype_ipw != "parametrized":
        pdos_list = tiptype_ipw
        path = os.path.dirname(hrstm_code.get_remote_exec_path()) + "/hrstm_tips/"
        pdos_list = [path + "tip_coeffs.tar.gz"]
        tip_pos = [ppm_qk + "PPpos", ppm_qk + "PPdisp"]
    else:  # Parametrized tip.
        pdos_list = [str(stip_ipw), str(pytip_ipw), str(pztip_ipw), str(pxtip_ipw)]
        tip_pos = ppm_qk + "PPdisp"

    # HRSTM parameters.
    paramdata = {
        "--output": "hrstm",
        "--voltages": [
            str(val)
            for val in np.round(
                np.arange(volmin_ipw, volmax_ipw + volstep_ipw, volstep_ipw),
                len(str(volstep_ipwmin).split(".")[-1]),
            ).tolist()
        ],
        # Sample information.
        "--cp2k_input_file": parent_dir + "aiida.inp",
        "--basis_set_file": parent_dir + "BASIS_MOLOPT",
        "--xyz_file": parent_dir + "aiida.coords.xyz",
        "--wfn_file": parent_dir + "aiida-RESTART.wfn",
        "--hartree_file": parent_dir + "aiida-HART-v_hartree-1_0.cube",
        "--emin": str(volmin_ipw - 2.0 * fwhm_ipw),
        "--emax": str(volmax_ipw + 2.0 * fwhm_ipw),
        "--fwhm_sam": str(fwhm_ipw),
        "--dx_wfn": str(wfnstep_ipw),
        "--extrap_dist": str(extrap_ipw),
        "--wn": str(workfun_ipw),
        # Tip information
        "--pdos_list": pdos_list,
        "--orbs_tip": str(orbstip_ipw),
        "--tip_shift": str(ppm_params_dict["rC0"][2] + ppm_params_dict["rO0"][2]),
        "--tip_pos_files": tip_pos,
        "--fwhm_tip": str(fwhmtip_ipw),
    }
    if rotate_ipw:
        paramdata["--rotate"] = ""
    return paramdata
