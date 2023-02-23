import ipywidgets as ipw


def get_start_widget(appbase, jupbase):
    # http://fontawesome.io/icons/
    template = """
    <table>
    <tr>
        <th style="text-align:center">Calculations</th>
        <th style="width:60px" rowspan=2></th>
        <th style="text-align:center">NEB and Replica chain</th>
    </tr>

    <tr>

    <td valign="top"><ul>
        <li><a href="{appbase}/submit_calculations.ipynb" target="_blank">Submit optimizatons</a>
        <li><a href="{appbase}/submit_adsorption_energy.ipynb" target="_blank">Compute adsorption energy</a>
        <li><a href="{appbase}/submit_gw.ipynb" target="_blank">Submit GW</a>
        <li><a href="{appbase}/submit_gw-ic.ipynb" target="_blank">Submit GW-IC</a>
        <li><a href="{appbase}/search.ipynb" target="_blank">Search</a>
    </ul></td>

    <td valign="top"><ul>
        <li><a href="{appbase}/submit_reactions.ipynb" target="_blank">Submit MEP</a>
    </ul></td>

    </tr>

    <tr>
        <th style="text-align:center">General</th>
        <th style="width:50px" rowspan=2></th>
        <th style="text-align:center">STM and PDOS</th>
        <th style="width:50px" rowspan=2></th>
        <th style="text-align:center">Viewers</th>
        <th style="width:50px" rowspan=2></th>
    </tr>

    <tr>
        <td valign="top"><ul>
            <li><a href="{appbase}/setup_codes.ipynb" target="_blank">Setup codes</a>
            <li><a href="{appbase}/manage_calcs.ipynb" target="_blank">Manage calculations</a>
        </ul></td>

        <td valign="top"><ul>
            <li><a href="{appbase}/submit_stm.ipynb" target="_blank">Submit STM</a>
            <li><a href="{appbase}/submit_pdos.ipynb" target="_blank">Submit PDOS</a>
        </ul></td>

        <td valign="top"><ul>
            <li><a href="{appbase}/view_pdos.ipynb" target="_blank">View PDOS</a>
            <li><a href="{appbase}/view_stm.ipynb" target="_blank">View STM</a>
            <li><a href="{appbase}/view_afm.ipynb" target="_blank">View AFM</a>
            <li><a href="{appbase}/view_hrstm.ipynb" target="_blank">View HR-STM</a>
            <li><a href="{appbase}/view_orb.ipynb" target="_blank">View ORB</a>
        </ul></td>


    </tr>

    </table>

"""

    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)


# EOF
