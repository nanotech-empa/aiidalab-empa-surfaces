import ipywidgets as ipw

def get_start_widget(appbase, jupbase):
    #http://fontawesome.io/icons/
    template = """
    <table>
    <tr>
        <th style="text-align:center">Structures</th>
        <th style="width:70px" rowspan=2></th>
        <th style="text-align:center">Nanoribbons</th>
        <th style="width:70px" rowspan=2></th>
        <th style="text-align:center">Slab Models</th>
        
    <tr>
    <td valign="top"><ul>
    <li><a href="{appbase}/upload_structure.ipynb" target="_blank">Upload structures</a>
    <li><a href="{appbase}/rescale_structure.ipynb" target="_blank">Scale structures</a>
    <li><a href="{appbase}/construct_cell.ipynb" target="_blank">Construct cell</a>
    <li><a href="{appbase}/edit_structure.ipynb" target="_blank">Assign spin, remove atoms</a>
    </ul></td>
    
    <td valign="top"><ul>
    <li><a href="{appbase}/nanoribbon/submit.ipynb" target="_blank">Submit calculation</a>
    <li><a href="{appbase}/nanoribbon/search.ipynb" target="_blank">Search database</a>
    </ul></td>
    
    <td valign="top"><ul>
    <li><a href="{appbase}/slab/build.ipynb" target="_blank">Construct slab</a>
    <li><a href="{appbase}/slab/submit_geopt.ipynb" target="_blank">Submit geo-opt</a>
    <li><a href="{appbase}/slab/search.ipynb" target="_blank">Search database</a>
    <li><a href="{appbase}/slab/submit_adsorption.ipynb" target="_blank">Adsorption energy</a>
    </ul></td>
    </tr></table>
"""
    
    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)
    
#EOF
