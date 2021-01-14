import sqlite3
import numpy as np
import io

from flask import current_app, g

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

def get_connection():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE_URI'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db


def close_connection(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()
