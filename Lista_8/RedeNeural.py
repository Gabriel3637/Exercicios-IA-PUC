import numpy as np
import random as rd

class RedeNeural:
    def __init__(self, num_camadas, num_neuronioscadacamada: list, taxa_aprendizado: float, num_entradas: int):
        self.num_camadas = num_camadas
        self.num_neuronioscadacamada = num_neuronioscadacamada
        self.num_entradas = num_entradas
        self.biasParaCadaNeuronio = []
        self.pesosBias = []
        self.entradas = []
        self.saidas = []
        self.pesos = []
        self.resultadosPorCamada = []
        self.camadas = []
        self.erros = []
        self.taxa_aprendizado = taxa_aprendizado
        for i in range(num_camadas):
            num_neuronios = num_neuronioscadacamada[i]
            if(i == 0):
                num_neuronios_camada_anterior = num_entradas
            else:
                num_neuronios_camada_anterior = num_neuronioscadacamada[i - 1]
            self.pesos.append(np.zeros((num_neuronios_camada_anterior, num_neuronios)))
            self.pesosBias.append(np.zeros((1, num_neuronios)))
            self.biasParaCadaNeuronio.append(np.ones((1, num_neuronios)))
            for j in range(num_neuronios):
                for k in range(num_neuronios_camada_anterior):
                    self.pesos[i][k][j] = rd.uniform(-1, 1)
                self.pesosBias[i][0][j] = rd.uniform(-1, 1)

            self.resultadosPorCamada.append(np.zeros((1, num_neuronios)))
            self.camadas.append(np.zeros((1, num_neuronios)))

    def atribuirPesos(self, lista_pesos: list):
        if len(lista_pesos) != self.num_camadas:
            raise ValueError("Número de matrizes de pesos deve ser igual ao número de camadas menos um.")
        self.pesos = lista_pesos
    def atribuirEntradas(self, entradas: np.ndarray):
        if(entradas.shape[1] != self.num_entradas):
            raise ValueError("Número de entradas deve ser igual ao número de neurônios na camada de entrada.")
        self.entradas = entradas
    def atribuirPesosBias(self, lista_pesos_bias: np.ndarray):
        if len(lista_pesos_bias) != self.num_camadas:
            raise ValueError("Número de matrizes de pesos bias deve ser igual ao número de camadas.")
        for i in range(self.num_camadas):
            if lista_pesos_bias[i].shape[1] != self.num_neuronioscadacamada[i]:
                raise ValueError(f"Número de colunas da matriz de pesos bias da camada {i} deve ser igual ao número de neurônios nessa camada.")
        self.pesosBias = lista_pesos_bias
    def atribuirBiasParaCadaNeuronio(self, lista_bias_para_cada_neuronio: list):
        if len(lista_bias_para_cada_neuronio) != self.num_camadas:
            raise ValueError("Número de vetores de bias deve ser igual ao número de camadas.")
        for i in range(self.num_camadas):
            if lista_bias_para_cada_neuronio[i].shape[1] != self.num_neuronioscadacamada[i]:
                raise ValueError(f"Número de colunas do vetor de bias da camada {i} deve ser igual ao número de neurônios nessa camada.")
        self.biasParaCadaNeuronio = lista_bias_para_cada_neuronio

    def aplicarFuncaoAtivacao(self, somatorio: np.ndarray):
        resp = 1 / (1 + np.exp(-somatorio))
        return resp

    def somatorio(self, entrada: np.ndarray, camada_atual: int):
        resp = entrada
        resp = resp @ self.pesos[camada_atual]
        resp = resp + self.biasParaCadaNeuronio[camada_atual] * self.pesosBias[camada_atual]
        return resp
    
    def resultado(self):
        entrada = self.entradas
        for i in range(self.num_camadas):
            self.resultadosPorCamada[i] = self.aplicarFuncaoAtivacao(self.somatorio(entrada, i))
            entrada = self.resultadosPorCamada[i]
        self.saidas = self.resultadosPorCamada[-1]
        return self.saidas
    
    def erroCamadaSaida(self, saida_esperada: np.ndarray):
        resp = None
        resp = (self.resultadosPorCamada[-1] * (1 - self.resultadosPorCamada[-1])) * (saida_esperada - self.resultadosPorCamada[-1])
        self.erros.insert(0, resp)
        return resp
    
    def erroPorCamada(self, saida_esperada: np.ndarray):
        self.erroCamadaSaida(saida_esperada)
        resp = None
        for camada_atual in range(self.num_camadas - 2, -1, -1):
            resp = (self.resultadosPorCamada[camada_atual] * (1 - self.resultadosPorCamada[camada_atual])) * (self.erros[0] @ self.pesos[camada_atual + 1].T)
            self.erros.insert(0, resp)
        return resp
    
    def ajustarPesos(self):
        for camada_atual in range(self.num_camadas):
            matrizPesosAtuais = self.pesos[camada_atual]
            for i in range(matrizPesosAtuais.shape[1]):
                ajusteBias = self.pesosBias[camada_atual][0][i] + (self.taxa_aprendizado * self.erros[camada_atual][0][i] * self.biasParaCadaNeuronio[camada_atual][0][i])
                self.pesosBias[camada_atual][0][i] = ajusteBias
                for j in range(matrizPesosAtuais.shape[0]):
                    entrada = None
                    if camada_atual == 0:
                        entrada = self.entradas[0][j]
                    else:
                        entrada = self.resultadosPorCamada[camada_atual - 1][0][j]
                    ajuste = matrizPesosAtuais[j][i] + (self.taxa_aprendizado * self.erros[camada_atual][0][i] * entrada)
                    matrizPesosAtuais[j][i] = ajuste

    def treinar(self, entradas: list, saida_esperada: list, epocas: int):
        for epoca in range(epocas):
            print(f"EPOCA {epoca+1}")
            for i in range(len(entradas)):
                self.atribuirEntradas(entradas[i])
                self.resultado()
                self.erroPorCamada(saida_esperada[i])
                self.ajustarPesos()

    def prever(self, entradas: list):
        resp = []
        for entrada in entradas:
            if entrada.shape[1] != self.num_entradas:
                raise ValueError("Número de entradas deve ser igual ao número de neurônios na camada de entrada.")
            self.atribuirEntradas(entrada)
            resultado = self.resultado()
            print(resultado)
            for i in range(resultado.shape[0]):
                for j in range(resultado.shape[1]):
                    if resultado[i][j] >= 0.5:
                        resultado[i][j] = 1
                    else:
                        resultado[i][j] = 0
            resp.append(resultado)

        return resp