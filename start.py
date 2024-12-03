import ipywidgets as ipw


import ipywidgets as ipw

def get_start_widget(appbase, jupbase):
    # Template for displaying icons with links
    template = f"""
    <table>
    <tr>
        <th style="text-align:center">Density Functional Theory</th>
        <th style="width:60px" rowspan=2></th>
        <th style="text-align:center">Post-processing</th>
    </tr>

    <tr>
    <td valign="top">
        <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
            <a href="{appbase}/submit_geometry_optimization.ipynb" target="_blank">
                <img src="{appbase}/buttons/geo_opt.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="Geometry Optimization">
                <div>Geometry Optimization</div>
            </a>
            <a href="{appbase}/submit_adsorption_energy.ipynb" target="_blank">
                <img src="{appbase}/buttons/ads_ene.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="Adsorption Energy">
                <div>Adsorption Energy</div>
            </a>
            <a href="{appbase}/submit_phonons.ipynb" target="_blank">
                <img src="{appbase}/buttons/phonons.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="Phonons">
                <div>Phonons</div>
            </a>
            <a href="{appbase}/submit_reactions.ipynb" target="_blank">
                <img src="{appbase}/buttons/ts_search.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="Transition State Search">
                <div>Transition State Search</div>
            </a>
            <a href="{appbase}/search.ipynb" target="_blank">
                <img src="{appbase}/buttons/search.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="Search">
                <div>Search</div>
            </a>
        </div>
    </td>

    <td valign="top">
        <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
            <a href="{appbase}/submit_spm.ipynb" target="_blank">
                <img src="{appbase}/buttons/spm.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="Scanning Probe Microscopy">
                <div>Scanning Probe Microscopy</div>
            </a>
            <a href="{appbase}/submit_pdos.ipynb" target="_blank">
                <img src="{appbase}/buttons/pdos.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="Projected Density of States">
                <div>Projected Density of States</div>
            </a>
        </div>
    </td>
    </tr>
    </table>
    """

    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)

