import eventlet
eventlet.monkey_patch()
from eventlet import wsgi

import sys
import logging

from flask import Flask
app = Flask(__name__)

from face_server import db
from face_server import rpc
from face_server import service

def main(argv):
    app.config.from_pyfile('face.cfg')
    app.teardown_appcontext(db.close_connection)
    rpc.init(app)

    if app.config['DEBUG']:
        app.run(port=8000, debug=True)
    else:
        wsgi.server(eventlet.listen(('', 8000)), app)


def run_main():  # pylint: disable=invalid-name
    try:
        sys.exit(main(sys.argv))
    except Exception as e:
        logging.exception('face server crashed...')
        sys.exit(1)
