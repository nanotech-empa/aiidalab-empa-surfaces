import ipywidgets as ipw

def get_start_widget(appbase, jupbase):
    #http://fontawesome.io/icons/
    template = """
    <table>
    <tr>
        <th style="text-align:center">General</th>
        <th style="width:70px" rowspan=2></th>
    </tr>
    
    <tr>
        <td valign="top"><ul>
            <li><a href="{appbase}/submit_geoopt.ipynb" target="_blank">Submit Geo Opt</a>
            <li><a href="{appbase}/submit_gw.ipynb" target="_blank">Submit GW or GW+IC</a>
            <li><a href="{appbase}/search.ipynb" target="_blank">Search Optimized molecules</a>
        </ul></td>
        

    </tr>
    </table>
"""
    
    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)
    
#EOF
