import ipywidgets as ipw


def get_start_widget(appbase, jupbase):
    # http://fontawesome.io/icons/
    template = """
    <table>
    <tr>
        <th style="text-align:center">DFT Calculations</th>
        <th style="width:60px" rowspan=2></th>
        <th style="text-align:center">SPM Calculations</th>
        <th style="text-align:center">GW calculations</th>
        <th style="text-align:center">Setup</th>        
    </tr>

    <tr>

    <td valign="top"><ul>
        <li><a href="{appbase}/submit_calculations.ipynb" target="_blank">Submit optimizatons</a>
        <li><a href="{appbase}/submit_adsorption_energy.ipynb" target="_blank">Compute adsorption energy</a>
        <li><a href="{appbase}/submit_phonons.ipynb" target="_blank">Submit Phonons</a>        
        <li><a href="{appbase}/submit_reactions.ipynb" target="_blank">Submit MEP</a>
        <li><a href="{appbase}/search.ipynb" target="_blank">Search</a>
    </ul></td>

    <td valign="top"><ul>
        <li><a href="{appbase}/submit_stm.ipynb" target="_blank">Submit STM</a>
        <li><a href="{appbase}/submit_pdos.ipynb" target="_blank">Submit PDOS</a>
    </ul></td>

    <td valign="top"><ul>
        <li><a href="{appbase}/submit_gw.ipynb" target="_blank">Submit GW</a>
        <li><a href="{appbase}/submit_gw-ic.ipynb" target="_blank">Submit GW-IC</a>    

    </ul></td>

    <td valign="top"><ul>
            <li><a href="{appbase}/setup_codes.ipynb" target="_blank">Setup SPM codes</a>
            <li><a href="{appbase}/manage_calcs.ipynb" target="_blank">Manage calculations</a>
    </ul></td>

    </tr>

    </table>

"""

    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)


# EOF
