import ipywidgets as ipw


def get_start_widget(appbase, jupbase):
    # http://fontawesome.io/icons/
    template = """
    <table>
    <tr>
        <th style="text-align:center">Density functional theory</th>
        <th style="width:60px" rowspan=2></th>
        <th style="text-align:center">Post-processing</th>
        <!--
        <th style="width:60px" rowspan=2></th>
        <th style="text-align:center">GW</th>
        -->
    </tr>

    <tr>

    <td valign="top"><ul>
        <li><a href="{appbase}/submit_geometry_optimization.ipynb" target="_blank">Geometry optimization</a>
        <li><a href="{appbase}/submit_adsorption_energy.ipynb" target="_blank">Adsorption energy</a>
        <li><a href="{appbase}/submit_phonons.ipynb" target="_blank">Phonons</a>
        <li><a href="{appbase}/submit_replica_chain.ipynb" target="_blank">Replica chain</a>
        <li><a href="{appbase}/submit_neb.ipynb" target="_blank">Nudged elastic band</a>
        <li><a href="{appbase}/submit_benchmark.ipynb" target="_blank">CSCS CP2K benchmark</a>
        <li><a href="{appbase}/search.ipynb" target="_blank">Search</a>
    </ul></td>

    <td valign="top"><ul>
        <li><a href="{appbase}/submit_spm.ipynb" target="_blank">Scanning probe microscopy</a>
        <li><a href="{appbase}/submit_pdos.ipynb" target="_blank">Projected density of states</a>
        <li><a href="{appbase}/handle_cubes.ipynb" target="_blank">Handle cube files</a>
    </ul></td>

    <!--
    <td valign="top"><ul>
        <li><a href="{appbase}/submit_gw.ipynb" target="_blank">GW</a>
        <li><a href="{appbase}/submit_gw_ic.ipynb" target="_blank">GW-IC</a>
    </ul></td>
    -->

    </tr>

    </table>

    """

    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html + get_cdxml_editor_card(appbase))


def get_cdxml_editor_card(appbase):
    """Return the launcher card for the standalone CDXML editor."""
    return f"""
        <div style="max-width:1120px;margin:14px auto;padding:0 6px">
            <a href="{appbase}/cdxml_editor.ipynb" target="_blank"
               style="display:flex;align-items:center;gap:14px;max-width:390px;
                      padding:14px;border:1px solid #d8dee6;border-radius:7px;
                      color:#1f2933;text-decoration:none;background:#fff">
                <span style="display:flex;padding:8px;border-radius:7px;
                             background:#eef7f1;color:#2a8c55">
                    <svg viewBox="0 0 64 64" width="42" height="42" aria-hidden="true"
                         fill="none" stroke="currentColor" stroke-width="3"
                         stroke-linecap="round" stroke-linejoin="round">
                        <path d="M25 10L8 20v20l17 10 17-10V20z" />
                        <path d="M13 23v14M25 44l12-7M25 16l12 7" />
                        <path d="M36 49l3-10 17-17 6 6-17 17zM51 27l6 6" />
                    </svg>
                </span>
                <span><strong>CDXML editor</strong><br />
                    <span style="font-size:13px;color:#5b6673">
                        Import, clean up, and edit ChemDraw structures.
                    </span>
                </span>
            </a>
        </div>
    """
