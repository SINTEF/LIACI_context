from http.server import HTTPServer, SimpleHTTPRequestHandler


httpd = HTTPServer(('localhost', 8800), SimpleHTTPRequestHandler)

httpd.serve_forever()