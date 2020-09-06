"""
Module to display our visualizations interactively
inside a Notebook / Browser.

This file is largely based on https://github.com/jakevdp/mpld3/blob/master/mpld3/_display.py
Copyright (c) 2013, Jake Vanderplas.
It was adapted for pyLDAvis by Ben Mabey.
It was then adapted for Texthero.
"""

import json
import jinja2
from ._server import serve


# Our HTML template. We use jinja2
# to programmatically insert the
# data we want to visualize
# in the function data_to_html
# below.
HTML_TEMPLATE = jinja2.Template(
    r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
    <link href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css" rel="stylesheet">
</head>

<body>
    <div class="container">
        <div class="header">
            <h5 class="text-muted"></h3>
        </div>

        <div>
            <div id="tablediv"></div>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.5.1.js" type="text/javascript"></script>
    <script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js" type="text/javascript"></script>
    <script src="https://cdn.datatables.net/plug-ins/1.10.21/dataRender/ellipsis.js" type="text/javascript"></script>
    <script type="text/javascript">
        


        $(document).ready(function () {
            $("#tablediv").html({{ df_json }});
            var table = $("#tableID").DataTable({
                columnDefs: [ {
                targets: 0,
                render: $.fn.dataTable.render.ellipsis(260, true, true)
    } ]
            });
        });

    </script>
</body>

</html>
"""
)


def data_to_html(df):
    """
    Output HTML with embedded visualization
    of the DataFrame df.

    """
    template = HTML_TEMPLATE

    # Create JSON from DataFrame with correct classes/ID for visualization.
    df_json = json.dumps(
        df.to_html(
            classes='table table-hover" id = "tableID',
            index=False,
            justify="left",
            border=0,
        )
    )

    return template.render(df_json=df_json)


def _display_df_notebook(df):
    """
    Display visualization of DataFrame `df`
    in IPython notebook via the HTML display hook.

    Returns the IPython HTML rich display of the visualization.

    """
    # import here, in case users don't have requirements installed
    try:
        from IPython.display import HTML
    except:
        raise ValueError(
            "You do not appear do be inside"
            " a Jupyter Notebook. Set"
            " notebook=False to show the visualization."
        )

    html = data_to_html(df)

    return HTML(html)


def _display_df_browser(
    df, ip="127.0.0.1", port=8888,
):
    """
    Display visualization of DataFrame `df`
    in local browser.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to visualize.

    ip : string, default = '127.0.0.1'
        The ip address used for the local server

    port : int, default = 8888
        The port number to use for the local server. 
        If already in use,
        a nearby open port will be found.

    """

    html = data_to_html(df)

    serve(
        html, ip=ip, port=port,
    )
