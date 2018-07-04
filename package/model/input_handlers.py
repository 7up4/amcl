from .importing_modules import *
import sqlalchemy
from PyQt5.QtSql import QSqlDatabase, QSqlQuery


class InputHandler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def feature_classes(self):
        """Read classes for represented features"""
        pass

    @abstractmethod
    def data(self):
        """Retrieve data from the input source and return an object."""
        pass


class FSHandler(InputHandler):
    def __init__(self, file_address: str, delimiter: str, header_line: int, feature_classes_line: int,
                 na_values: list):
        self.__file = open(file_address)
        self.__delimiter = delimiter
        self.__header_line = header_line
        self.__feature_classes_line = feature_classes_line
        self.__na_values = na_values

    @property
    def feature_classes(self) -> list:
        self.__file.seek(0)
        for i in range(self.__feature_classes_line):
            self.__file.readline()
        return self.__file.readline().strip().split(self.__delimiter)

    @property
    def data(self) -> pd.DataFrame:
        self.__file.seek(0)
        return pd.read_csv(self.__file, delimiter=self.__delimiter, header=self.__header_line,
                           na_values=self.__na_values, nrows=303)

    @property
    def file(self):
        return self.__file

    @file.setter
    def file(self, file_address):
        self.__file = open(file_address)


class QtSqlDBHandler(InputHandler):
    def __init__(self, dbname: str, username: str, password: str, port, table_name, hostname: str = "127.0.0.1"):
        self.__db = QSqlDatabase.addDatabase("QPSQL")
        self.__table_name = table_name
        self.__hostname = hostname
        self.__dbname = dbname
        self.__port = port
        self.__username = username
        self.__password = password
        self.__status = False

    def configure(self):
        self.__db.setHostName(self.__hostname)
        self.__db.setDatabaseName(self.__dbname)
        self.__db.setUserName(self.__username)
        self.__db.setPassword(self.__password)
        self.__db.setPort(self.__port)

    def open(self):
        self.__status = self.__db.open()
        if not self.__status:
            print(self.__db.lastError().text())

    def close(self):
        self.__db.close()
        self.__status = False

    def data(self, *features) -> pd.DataFrame:
        request = "SELECT * FROM " + self.__table_name + " ;"
        query = QSqlQuery()
        if self.__status:
            query.setForwardOnly(True)
            query.exec(request)
            d = {el: [] for el in features}
            query.seek(0)
            while query.next():
                for i in features:
                    d[i].append(query.value(i))
            return pd.DataFrame(data=d)

    # Not implemented yet
    @property
    def future_classes(self):
        return


class SqlAlchemyDBHandler(InputHandler):
    def __init__(self, db_dialect, dbname: str, username: str, password: str, port, table_name,
                 hostname: str = "127.0.0.1"):
        self.__db = None
        self.__db_dialect = db_dialect
        self.__table_name = table_name
        self.__hostname = hostname
        self.__dbname = dbname
        self.__port = port
        self.__username = username
        self.__password = password
        self.__connection = None

    def configure(self):
        self.__db = sqlalchemy.create_engine("{}://{}:{}@{}:{}/{}".format(self.__db_dialect, self.__username,
                                                                          self.__password, self.__hostname,
                                                                          self.__port, self.__dbname))

    def open(self):
        self.__connection = self.__db.connect()

    def close(self):
        if self.__connection is not None:
            self.__connection.close()

    def data(self, *features) -> pd.DataFrame:
        if not features:
            return pd.read_sql_table(self.__table_name, self.__db)
        else:
            features = ",".join(features)
            return pd.read_sql_query("SELECT {} FROM {};".format(features, self.__table_name), self.__connection)

    # Not implemented yet
    @property
    def future_classes(self):
        return
