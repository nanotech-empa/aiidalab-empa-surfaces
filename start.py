import ipywidgets as ipw


import ipywidgets as ipw

def get_start_widget(appbase, jupbase):
    # Template for displaying icons with links
    template = f"""
    <table>
    <tr>
        <td valign="top" style="text-align:center;">
            <a href="{appbase}/submit_geometry_optimization.ipynb" target="_blank">
                <img src="{appbase}/buttons/geo_opt.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="Geometry Opt.">
                <div>Geometry Opt.</div>
            </a>
        </td>
        <td valign="top" style="text-align:center;">
            <a href="{appbase}/submit_adsorption_energy.ipynb" target="_blank">
                <img src="{appbase}/buttons/ads_ene.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="Adsorption Ene.">
                <div>Adsorption Ene.</div>
            </a>
        </td>
        <td valign="top" style="text-align:center;">
            <a href="{appbase}/submit_reactions.ipynb" target="_blank">
                <img src="{appbase}/buttons/ts_search.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="TS Search">
                <div>TS Search</div>
            </a>
        </td>
        <td valign="top" style="text-align:center;">
            <a href="{appbase}/submit_phonons.ipynb" target="_blank">
                <img src="{appbase}/buttons/phonons.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="Phonons">
                <div>Phonons</div>
            </a>
        </td>
    </tr>
    <tr>
        <td valign="top" style="text-align:center;" colspan="2">
            <a href="{appbase}/submit_spm.ipynb" target="_blank">
                <img src="{appbase}/buttons/spm.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="SPM">
                <div>SPM</div>
            </a>
        </td>
        <td valign="top" style="text-align:center;" colspan="2">
            <a href="{appbase}/submit_pdos.ipynb" target="_blank">
                <img src="{appbase}/buttons/pdos.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="PDOS">
                <div>PDOS</div>
            </a>
        </td>
    </tr>
    <tr>
        <td valign="top" style="text-align:center;" colspan="4">
            <a href="{appbase}/search.ipynb" target="_blank">
                <img src="{appbase}/buttons/search.png?raw=true" height="60px" width="120px" style="cursor: pointer;" alt="Search">
                <div>Search</div>
            </a>
        </td>
    </tr>
    </table>
    """

    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)


