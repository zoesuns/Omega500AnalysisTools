import pandas as pd

class return_tables :
    
    def __init__( database, table_keys=['halos','profiles',
                              'mergertree','mergers']) :

    import sqlite3
    connection = sqlite3.connect(database)

    return {table_key:
            pd.read_sql("SELECT * FROM "+table_key, connection) \
            for table_key in table_keys}
