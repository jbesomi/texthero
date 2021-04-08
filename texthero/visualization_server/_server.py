# this file is largely based on https://github.com/jakevdp/mpld3/blob/master/mpld3/_server.py
# Copyright (c) 2013, Jake Vanderplas
"""
Simple server used to serve our visualizations in a web browser.
"""
from http import server
import sys
import threading
import webbrowser
import socket


def generate_handler(html):
    """
    Generate handler that only
    serves our generated html.
    """

    class MyHandler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            """Respond to a GET request."""
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())
            else:
                self.send_error(404)

    return MyHandler


def find_open_port(ip, port, n=50):
    """
    Find an open port near the specified port.
    """

    ports = [port + i for i in range(n)]

    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((ip, port))
        s.close()
        if result != 0:
            return port

    raise ValueError("no open ports found")


def serve(
    html, ip="127.0.0.1", port=8888, open_browser=True,
):
    """
    Start a server serving the given HTML, and (optionally) open a
    browser.

    Parameters
    ----------
    html : string
        HTML to serve

    ip : string (default = '127.0.0.1')
        ip address at which the HTML will be served.

    port : int (default = 8888)
        the port at which to serve the HTML

    open_browser : bool (optional)
        if True (default), then open a web browser to the given HTML
    """

    port = find_open_port(ip, port, n=50)
    Handler = generate_handler(html)

    srvr = server.HTTPServer((ip, port), Handler)

    # Start the server
    print("Serving to http://{0}:{1}/    [Ctrl-C to exit]".format(ip, port))
    sys.stdout.flush()

    if open_browser:
        # Use a thread to open a web browser pointing to the server
        def b():
            return webbrowser.open("http://{0}:{1}".format(ip, port))

        threading.Thread(target=b).start()

    try:
        srvr.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        print("\nStopping Server...")

    srvr.server_close()
