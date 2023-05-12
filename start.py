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
        <li><a href="{appbase}/submit_calculations.ipynb" target="_blank">Geometry optimization</a>
        <li><a href="{appbase}/submit_adsorption_energy.ipynb" target="_blank">Adsorption energy</a>
        <li><a href="{appbase}/submit_phonons.ipynb" target="_blank">Phonons</a>
        <li><a href="{appbase}/submit_reactions.ipynb" target="_blank">Reactions</a>
        <li><a href="{appbase}/search.ipynb" target="_blank">Search</a>
    </ul></td>

    <td valign="top"><ul>
        <li><a href="{appbase}/submit_spm.ipynb" target="_blank">Scanning probe microscopy</a>
        <li><a href="{appbase}/submit_pdos.ipynb" target="_blank">Projected density of states</a>
    </ul></td>

    <!--
    <td valign="top"><ul>
        <li><a href="{appbase}/submit_gw.ipynb" target="_blank">GW</a>
        <li><a href="{appbase}/submit_gw-ic.ipynb" target="_blank">GW-IC</a>
    </ul></td>
    -->

    </tr>

    </table>

    """

    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)
