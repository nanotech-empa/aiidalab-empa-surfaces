import ipywidgets as ipw

def get_start_widget(appbase, jupbase):
    #http://fontawesome.io/icons/
    template = """
    <table>
    <tr>
        <th style="text-align:center">Calculations</th>
        <th style="width:60px" rowspan=2></th>
        <th style="text-align:center">Constr. opt. chains</th>
        <th style="width:60px" rowspan=2></th>
        <th style="text-align:center">Nudged elastic band</th>
    </tr>
    
    <tr>
    
    <td valign="top"><ul>
        <li><a href="{appbase}/submit_calculations.ipynb" target="_blank">Submit optimizatons</a>
        <li><a href="{appbase}/gw/submit_gw.ipynb" target="_blank">Submit GW</a>
        <li><a href="{appbase}/slab/build_slab.ipynb" target="_blank">Build slab</a>
        <li><a href="{appbase}/search.ipynb" target="_blank">Search</a>
    </ul></td>
    
    <td valign="top"><ul>
        <li><a href="{appbase}/reactions/submit_replicas.ipynb" target="_blank">Generate replicas</a>
        <li><a href="{appbase}/reactions/search_replicas.ipynb" target="_blank">Search replica chains</a>
    </ul></td>
    
    <td valign="top"><ul>
        <li><a href="{appbase}/reactions/submit_neb.ipynb" target="_blank">Submit NEB</a>
        <li><a href="{appbase}/reactions/search_neb.ipynb" target="_blank">Search NEBs</a>
    </ul></td>
    
    </tr></table>

"""
    
    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)
    
#EOF
