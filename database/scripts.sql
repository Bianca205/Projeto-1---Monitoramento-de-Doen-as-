-- Criar database
CREATE DATABASE IF NOT EXISTS monitoramento_saude;
USE monitoramento_saude;

-- Tabela de Pacientes
CREATE TABLE IF NOT EXISTS pacientes (
    id_paciente INT AUTO_INCREMENT PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    idade INT,
    sexo ENUM('M', 'F'),
    data_cadastro DATE,
    telefone VARCHAR(20),
    email VARCHAR(100)
);

-- Tabela de Exames
CREATE TABLE IF NOT EXISTS exames (
    id_exame INT AUTO_INCREMENT PRIMARY KEY,
    id_paciente INT,
    data_exame DATE,
    tipo_exame VARCHAR(50),
    valor_exame DECIMAL(10,2),
    unidade VARCHAR(20),
    FOREIGN KEY (id_paciente) REFERENCES pacientes(id_paciente)
);

-- Tabela de Diagnósticos
CREATE TABLE IF NOT EXISTS diagnosticos (
    id_diagnostico INT AUTO_INCREMENT PRIMARY KEY,
    id_paciente INT,
    data_diagnostico DATE,
    doenca VARCHAR(100),
    gravidade ENUM('leve', 'moderada', 'grave'),
    observacoes TEXT,
    FOREIGN KEY (id_paciente) REFERENCES pacientes(id_paciente)
);

-- Inserir dados de exemplo
INSERT INTO pacientes (nome, idade, sexo, data_cadastro) VALUES
('João Silva', 45, 'M', '2023-01-15'),
('Maria Santos', 52, 'F', '2023-02-20'),
('Pedro Oliveira', 38, 'M', '2023-03-10');

INSERT INTO exames (id_paciente, data_exame, tipo_exame, valor_exame, unidade) VALUES
(1, '2023-04-01', 'pressao_sistolica', 120, 'mmHg'),
(1, '2023-04-01', 'colesterol', 200, 'mg/dL'),
(2, '2023-04-02', 'pressao_sistolica', 140, 'mmHg'),
(2, '2023-04-02', 'colesterol', 240, 'mg/dL');