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
            <li><a href="{appbase}/upload_many_structures.ipynb" target="_blank">Upload Structures</a>
            <li><a href="{appbase}/submit_cellopt.ipynb" target="_blank">Submit Cell Opt</a>
            <li><a href="{appbase}/submit_bulkopt.ipynb" target="_blank">Submit Bulk Opt</a>
            <li><a href="{appbase}/search.ipynb" target="_blank">Search Cell Opt</a>
        </ul></td>
        

    </tr>
    </table>
"""
    
    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)
    
#EOF
