import mysql.connector
from mysql.connector import Error
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.host = os.getenv('DB_HOST')
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')
        self.database = os.getenv('DB_NAME')
    
    def create_connection(self):
        """Cria uma conexão com o banco de dados MySQL"""
        connection = None
        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            print("✅ Conexão com MySQL estabelecida")
        except Error as e:
            print(f"❌ Erro ao conectar ao MySQL: {e}")
        return connection

    def execute_query(self, query, params=None):
        """Executa uma query (INSERT, UPDATE, DELETE)"""
        connection = self.create_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(query, params or ())
            connection.commit()
            print("✅ Query executada com sucesso")
        except Error as e:
            print(f"❌ Erro ao executar query: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def read_query(self, query, params=None):
        """Executa uma query de leitura (SELECT) e retorna um DataFrame"""
        connection = self.create_connection()
        try:
            df = pd.read_sql(query, connection, params=params)
            return df
        except Error as e:
            print(f"❌ Erro ao ler dados: {e}")
            return pd.DataFrame()
        finally:
            if connection.is_connected():
                connection.close()

# Instância global
db = DatabaseManager()