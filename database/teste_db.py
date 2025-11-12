from database import db

def test_database():
    # Testar conexão
    connection = db.create_connection()
    if connection:
        print("✅ Conexão estabelecida com sucesso!")
        connection.close()
    
    # Testar leitura de tabelas
    try:
        tables = db.read_query("SHOW TABLES")
        print(f"📊 Tabelas no banco: {len(tables)}")
        for table in tables.values:
            print(f"  - {table[0]}")
            
        # Inserir um paciente de teste
        insert_query = """
        INSERT INTO pacientes (nome, data_nascimento, sexo) 
        VALUES (%s, %s, %s)
        """
        db.execute_query(insert_query, ('Paciente Teste', '1990-01-01', 'M'))
        
        # Ver pacientes
        pacientes = db.read_query("SELECT * FROM pacientes")
        print(f"\n📝 Pacientes cadastrados: {len(pacientes)}")
        print(pacientes)
        
    except Exception as e:
        print(f"❌ Erro nos testes: {e}")

if __name__ == "__main__":
    test_database()