import pandas as pd
import treelib as tl
import math
import numpy as np
import copy
from itertools import combinations

class No:
    def __init__(self, nomeRegra: str, entropia: float, registros: int, classesPrev: list, implementacaoRegra, classe: str | None = None, atributo: str | None = None):
        self.nomeRegra = nomeRegra
        self.atributo = atributo
        self.implementacaoRegra = implementacaoRegra
        self.entropia = entropia
        self.registros = registros
        self.classesPrev = classesPrev
        self.classe = classe
    def print(self):
        print(f"Regra: {self.nomeRegra}")
        if self.classe is not None:
            print(f"Classe: {self.classe}")
        else:
            print(f"Atributo: {self.atributo}")
        print(f"Entropia: {self.entropia}")
        print(f"Registros: {self.registros}")
        print(f"ClassesPrev: {self.classesPrev}")
    def getClassesAtrib(self):
        return self.classesAtrib
    def getEntropia(self):
        return self.entropia
    def getRegistros(self):
        return self.registros
    def getClassesPrev(self):
        return self.classesPrev
    def getClasse(self):
        return self.classe
    def getAtributo(self):
        return self.atributo
    def getNomeRegra(self):
        return self.nomeRegra
    def getGanho(self):
        return self.ganho
    def testImplementacaoRegra(self, valor):
        return self.implementacaoRegra(valor)
    def printAll(self):
        self.print()
    
    def __str__(self):
        return f"Regra: {self.nomeRegra} | Atributo: {self.atributo} | Entropia: {self.entropia} | Registros: {self.registros} | ClassesPrev: {self.classesPrev} | ImplementacaoRegra: {self.implementacaoRegra}"
    def __repr__(self):
        return self.__str__()




class Categoria:
    def __init__(self, Nome: str, entropia: float, ganho: float, registros: int, classesPrev: list, valoresAtributos: list = list()):
        self.Nome = Nome
        self.entropia = entropia
        self.ganho = ganho
        self.registros = registros
        self.classesPrev = classesPrev
        self.valoresAtributos = valoresAtributos
    def print(self):
        print(f"Nome: {self.Nome}")
        print(f"Entropia: {self.entropia}")
        print(f"Registros: {self.registros}")
        print(f"ClassesPrev: {self.classesPrev}")
    def getValoresAtributos(self):
        return self.valoresAtributos
    def getEntropia(self):
        return self.entropia
    def getRegistros(self):
        return self.registros
    def getClassesPrev(self):
        return self.classesPrev
    def getNome(self):
        return self.Nome
    def getGanho(self):
        return self.ganho
    def setGanho(self, ganho: float):
        self.ganho = ganho
        return self
    def setValoresAtributos(self, valoresAtributos: list):
        self.valoresAtributos = valoresAtributos
        return self
    def setEntropia(self, entropia: float):
        self.entropia = entropia
        return self 
    def setRegistros(self, registros: int):
        self.registros = registros
        return self
    def setClassesPrev(self, classesPrev: list):
        self.classesPrev = classesPrev
        return self
    def setNome(self, Nome: str):
        self.Nome = Nome
        return self
    def printAll(self):
        self.print()
        print(f"Ganho: {self.ganho}")
        print(f"ValoresAtributos: {self.valoresAtributos}")
    def __str__(self):
        return f"Nome: {self.Nome} \n Entropia: {self.entropia} \n Registros: {self.registros} \n ClassesPrev: {self.classesPrev} \n Ganho: {self.ganho} \n ValoresAtributos: {self.valoresAtributos}"
    def __repr__(self):
        return self.__str__() + "\n"


class ArvoreCART:
    def __init__(self):
        self.arvore = tl.Tree()

    def __calcular_gini_classe(self, coluna : pd.DataFrame):
        distribuicaoValoresDiferentes = coluna.value_counts()
        quantidadeRegistros = len(coluna)
        probabilidadeQuadratica = 0
        gini = 1
        for valor, quantidade in distribuicaoValoresDiferentes.items():
            probabilidadeQuadratica  += (math.pow((quantidade/quantidadeRegistros), 2))

        return gini - probabilidadeQuadratica 
    def __calcular_gini_atributo(self, atributo: pd.DataFrame, classe: pd.DataFrame, ):
        distribuicaoValoresDiferentesAtrib = atributo.value_counts()

        concatenacaoAtribClass = pd.concat([atributo, classe], axis=1)
        concatenacaoAtribClass = concatenacaoAtribClass.dropna(axis=0, how='any', subset=[atributo.columns[0]])
        distribuicaoValoresDiferentesClasse = concatenacaoAtribClass[classe.columns[0]].value_counts()
        quantidadeRegistros = len(atributo)
        probabilidadeQuadratica = 0
        giniValor = 1
        gini = 0
        testeAtributo = 0
        quantidadeClasseporAtributo = 0
        listaGinis = list()
        quantidadeResultadoClasse = list()
        for valorAtributo, quantidadeAtributo in distribuicaoValoresDiferentesAtrib.items():
            for valorClasse, quantidadeClasse in distribuicaoValoresDiferentesClasse.items():
                quantidadeClasseporAtributo = len(concatenacaoAtribClass[(concatenacaoAtribClass[atributo.columns[0]] == valorAtributo[0]) & (concatenacaoAtribClass[classe.columns[0]] == valorClasse)])
                quantidadeResultadoClasse.append((valorClasse,quantidadeClasseporAtributo))
                if(quantidadeClasseporAtributo > 0):
                    probabilidadeQuadratica += (math.pow((quantidadeClasseporAtributo/quantidadeAtributo), 2))
            giniValor -= probabilidadeQuadratica
            probabilidadeQuadratica = 0
            gini += (giniValor * (quantidadeAtributo/quantidadeRegistros))
            listaGinis.append((valorAtributo[0], giniValor, quantidadeResultadoClasse.copy()))
            giniValor = 1
            quantidadeResultadoClasse.clear()
        return gini, list(distribuicaoValoresDiferentesClasse.items()), listaGinis

    def __gerarMascara(self, registros, filtros):
        mascara = pd.Series([True] * len(registros))
        for filtro in filtros:
            if(filtro[2] == '=='):
                mascara = mascara & (registros[filtro[0]] == filtro[1])
            elif(filtro[2] == '!='):
                mascara = mascara & (registros[filtro[0]] != filtro[1])
            elif(filtro[2] == 'in'):
                mascara = mascara & (registros[filtro[0]].isin(filtro[1]))
            elif(filtro[2][0] == '>'):
                if(filtro[2] == '>='):
                    mascara = mascara & (registros[filtro[0]] >= filtro[1])
                else:
                    mascara = mascara & (registros[filtro[0]] > filtro[1])

            elif(filtro[2][0] == '<'):
                if(filtro[2] == '<='):
                    mascara = mascara & (registros[filtro[0]] <= filtro[1])
                else:
                    mascara = mascara & (registros[filtro[0]] < filtro[1])
        return mascara
    
    def __continuos_maior_razao_ganho(self, atributos: pd.DataFrame, classe: pd.DataFrame, coluna: str, filtros: list = list()):
        resp = None
        tmp = None
        giniClasse = self.__calcular_gini_classe(classe)
        valoresUnicos = sorted(atributos[coluna].unique())
        if(len(valoresUnicos) > 1):
            pontosCorte = [(valoresUnicos[i] + valoresUnicos[i+1])/2 for i in range(len(valoresUnicos)-1)]
            for pontoCorte in pontosCorte:
                atributoBinarizado = atributos[[coluna]].map(lambda x: f"<= {pontoCorte}" if x <= pontoCorte else f"> {pontoCorte}")
                entropiaAtributo, registrosClasses, listaEntropias = self.__calcular_gini_atributo(atributoBinarizado, classe)
                for indice, valor in enumerate(listaEntropias):
                    if(valor[0][0] == '<'):
                        listaEntropias[indice] = (float(valor[0][3:]), '<=', valor[1], valor[2])
                    else:
                        listaEntropias[indice] = (float(valor[0][2:]), '>', valor[1], valor[2])
                tmp = Categoria(coluna, entropiaAtributo, (giniClasse - entropiaAtributo), sum(item[1] for item in registrosClasses), registrosClasses, listaEntropias)
                if(resp is None or tmp.getGanho() > resp.getGanho()):
                    resp = copy.deepcopy(tmp)
        else:
            entropiaAtributo, registrosClasses, listaEntropias = self.__calcular_gini_atributo(atributos[[coluna]], classe)
            for indice, valor in enumerate(listaEntropias):
                listaEntropias[indice] = (float(valor[0]), '==', valor[1], valor[2])
            
            resp = Categoria(coluna, entropiaAtributo, (giniClasse - entropiaAtributo), sum(item[1] for item in registrosClasses), registrosClasses, listaEntropias)

        return resp
    
    def dividir_em_dois_conjuntos(lista):
        tamanho_total = len(lista)
        if(tamanho_total >= 2):
            valorFinalDoRange = tamanho_total // 2 + 1
        else:
            valorFinalDoRange = tamanho_total + 1
        for i in range(1, valorFinalDoRange):
            for conjunto1_comb in combinations(lista, i):

                conjunto1 = list(conjunto1_comb)
                
                lista_original_set = set(lista)
                conjunto1_set = set(conjunto1)
                

                conjunto2 = list(lista_original_set.difference(conjunto1_set))
                
                yield (conjunto1, conjunto2)
        
    def __categoricos_maior_reducao_gini(self, atributos: pd.DataFrame, classe: pd.DataFrame, coluna: str, filtros: list = list()):
        resp = None
        tmp = None
        giniClasse = self.__calcular_gini_classe(classe)
        valoresUnicos = sorted(atributos[coluna].unique())

        for conjunto1, conjunto2 in ArvoreCART.dividir_em_dois_conjuntos(valoresUnicos):
            atributoBinarizado = atributos[[coluna]].map(lambda x: f"GRUPO1" if x in conjunto1 else f"GRUPO2")
            giniAtributo, registrosClasses, listaGinis = self.__calcular_gini_atributo(atributoBinarizado, classe)
            for indice, valor in enumerate(listaGinis):
                if(valor[0] == 'GRUPO1'):
                    listaGinis[indice] = (conjunto1, 'in', valor[1], valor[2])
                else:
                    listaGinis[indice] = (conjunto2, 'in', valor[1], valor[2])
            tmp = Categoria(coluna, giniAtributo, (giniClasse - giniAtributo), sum(item[1] for item in registrosClasses), registrosClasses, listaGinis)
            if(resp is None or tmp.getGanho() > resp.getGanho()):
                resp = copy.deepcopy(tmp)

        return resp



            
        

    def __atributo_maior_reducao_gini(self, atributos: pd.DataFrame, classe: pd.DataFrame, filtros: list = list()):
        resp = None
        tmp = None
        mascara = self.__gerarMascara(atributos, filtros)
        registrosFiltrados = atributos[mascara]
        classesFiltradas = classe[mascara]
        giniClasse = self.__calcular_gini_classe(classesFiltradas)
        for coluna in registrosFiltrados.columns:
            if not any((filtro[0] == coluna and (filtro[2] == "in" or filtro[2] == "==")) for filtro in filtros):
                if(pd.api.types.is_numeric_dtype(registrosFiltrados[coluna]) and not pd.api.types.is_bool_dtype(registrosFiltrados[coluna])):
                    tmp = self.__continuos_maior_razao_ganho(registrosFiltrados, classesFiltradas, coluna, filtros)
                else:
                    tmp = self.__categoricos_maior_reducao_gini(registrosFiltrados, classesFiltradas, coluna, filtros)
                if(resp is None or tmp.getGanho() > resp.getGanho()):
                    resp = copy.deepcopy(tmp)


        return resp

    def gerar_test_treino(self, atributosGerais: pd.DataFrame, atributosPrevisao: pd.DataFrame, valorAleatorização: int, tamanhoTeste: float):
        if tamanhoTeste >= 1 and atributosGerais.shape[0] != atributosPrevisao.shape[0]:
            raise ValueError("O tamanho do teste deve ser inferior a 1")
        else: 
            quantidadeRegistros = len(atributosGerais)
            quantidadeTeste = math.floor(quantidadeRegistros * tamanhoTeste)
            
            atributosGeraisAleatorio = atributosGerais.sample(frac=1, random_state=valorAleatorização).reset_index(drop=True)
            atributosPrevisaoAleatorio = atributosPrevisao.sample(frac=1, random_state=valorAleatorização).reset_index(drop=True)

            x_test = atributosGeraisAleatorio.tail(quantidadeTeste)
            y_test = atributosPrevisaoAleatorio.tail(quantidadeTeste)
            x_treino = atributosGeraisAleatorio.head(quantidadeRegistros - quantidadeTeste)
            y_treino = atributosPrevisaoAleatorio.head(quantidadeRegistros - quantidadeTeste)
            
        return x_treino, y_treino, x_test, y_test
    
    
    def __create_node(self, tag, identifier, parent:int | None = None, data: No | None = None):
        self.arvore.create_node(tag=tag, identifier=identifier, parent=parent, data=data)
    def show(self):
        self.arvore.show()

    def __gerarRegraLambda(self, operacao, coluna, valor):
        regraLambda = None
        regraString = None
        if(operacao == '=='):
            regraLambda = lambda registro, colunal=coluna, valorl=valor: registro[colunal] == valorl
            regraString = f'{coluna} == {valor}'
        elif(operacao == '!='):
            regraLambda = lambda registro, colunal=coluna, valorl=valor: registro[colunal] != valorl
            regraString = f'{coluna} != {valor}'

        elif(operacao == 'in'):
            regraLambda = lambda registro, colunal=coluna, valorl=valor: registro[colunal] in valorl
            regraString = f'{coluna} in {valor}'
        elif(operacao[0] == '>'):
            if(operacao == '>='):
                regraLambda = lambda registro, colunal=coluna, valorl=valor: registro[colunal] >= valorl
                regraString = f'{coluna} >= {valor}'
            else:
                regraLambda = lambda registro, colunal=coluna, valorl=valor: registro[colunal] > valorl
                regraString = f'{coluna} > {valor}'
        elif(operacao[0] == '<'):
            if(operacao == '<='):
                regraLambda = lambda registro, colunal=coluna, valorl=valor: registro[colunal] <= valorl
                regraString = f'{coluna} <= {valor}'
            else:
                regraLambda = lambda registro, colunal=coluna, valorl=valor: registro[colunal] < valorl
                regraString = f'{coluna} < {valor}'
        
        return regraLambda, regraString
    

    def gerar_arvore(self, x_treino: pd.DataFrame, y_treino: pd.DataFrame, tamanhoLimite: int = None):
        if(len(x_treino) > 0 and len(y_treino) > 0):
            for coluna in x_treino.columns:
                if pd.api.types.is_numeric_dtype(x_treino[coluna]) and not pd.api.types.is_bool_dtype(x_treino[coluna]):
                    media = x_treino[coluna].mean()
                    x_treino[coluna] = x_treino[coluna].fillna(media)
                else:
                    moda = x_treino[coluna].mode()[0]
                    x_treino[coluna] = x_treino[coluna].fillna(moda)


            baseXTreino = x_treino.copy()
            baseYTreino = y_treino.copy()
            paiID = 0
            countID = 0

            pilhaRegrasPendente = list()
            pilhaRegrasPai = list()
            controle = True
            
            while(controle):
                if pilhaRegrasPendente:
                    pilhaRegrasPai.append(pilhaRegrasPendente.pop())
                if pilhaRegrasPai:
                    paiID = pilhaRegrasPai[-1][2]
                    regrasPaiImplementacao, strRegrasPai = self.__gerarRegraLambda(pilhaRegrasPai[-1][2], pilhaRegrasPai[-1][0], pilhaRegrasPai[-1][1])
                else:
                    regrasPaiImplementacao = lambda registo: True
                    strRegrasPai = "True"

                atributo_atual = self.__atributo_maior_reducao_gini(baseXTreino, baseYTreino, pilhaRegrasPai)
                if(atributo_atual is None):
                    noAnterior = self.arvore.get_node(pilhaRegrasPai[-1][3])
                    termoClasse = max(noAnterior.data.getClassesPrev(), key= lambda item: item[1] )[0]
                    mascaraNoultimo = self.__gerarMascara(baseXTreino, pilhaRegrasPai)
                    registrosFiltrados = baseYTreino[mascaraNoultimo]
                    entropiaNoultimo = self.__calcular_gini_classe(registrosFiltrados)
                    self.__create_node(tag=f"{strRegrasPai} -> {termoClasse} Qtd: {noAnterior.data.getRegistros()}", identifier=countID, parent=noAnterior.identifier, data=
                                       No(strRegrasPai, entropiaNoultimo, len(registrosFiltrados), list(registrosFiltrados.value_counts().items()), regrasPaiImplementacao, termoClasse, noAnterior.data.getAtributo()))
                    quantidadeComConfusao = 0
                else:
                    No_atual = No(strRegrasPai, atributo_atual.entropia, atributo_atual.registros, atributo_atual.classesPrev, regrasPaiImplementacao, None, atributo_atual.getNome())

                    self.__create_node(tag=f"{strRegrasPai} -> Testar: {atributo_atual.getNome()} Qtd: {atributo_atual.getRegistros()} Entropia: {atributo_atual.getEntropia()}", identifier=countID, parent= None if countID==0 else pilhaRegrasPai[-1][3], data=No_atual)
                    quantidadeComConfusao = sum(1 for ent in atributo_atual.getValoresAtributos() if ent[2] > 0)

                    paiID = countID
                    dentroDoLimiteDeRegras = (len(pilhaRegrasPai) + 1) < tamanhoLimite if tamanhoLimite is not None else True
                    if(dentroDoLimiteDeRegras):
                        for entropia in atributo_atual.getValoresAtributos():
                            if entropia[2] > 0:
                                pilhaRegrasPendente.append((atributo_atual.getNome(), entropia[0], entropia[1], paiID))
                            else:
                                implementacaoRegraFilho, strRegrasFilho = self.__gerarRegraLambda(entropia[1], atributo_atual.getNome(), entropia[0])
                                countID += 1
                                termoClasse = max(entropia[3], key= lambda item: item[1] )[0]
                                self.__create_node(tag=f"{strRegrasFilho} -> {termoClasse} Qtd: {sum(item[1] for item in entropia[3])}", identifier=countID, parent=paiID, data=
                                                No(strRegrasFilho, 0, sum(item[1] for item in entropia[3]), entropia[3], implementacaoRegraFilho, termoClasse, atributo_atual.getNome()))
                    else:
                        for entropia in atributo_atual.getValoresAtributos():
                            implementacaoRegraFilho = lambda registro, coluna=atributo_atual.getNome(), valor=entropia[0]: registro[coluna] == valor
                            strRegrasFilho = f"{atributo_atual.getNome()} == {entropia[0]}"
                            countID += 1
                            termoClasse = max(entropia[3], key= lambda item: item[1] )[0]
                            self.__create_node(tag=f"{strRegrasFilho} -> {termoClasse} Qtd: {sum(item[1] for item in entropia[3])}", identifier=countID, parent=paiID, data=
                                            No(strRegrasFilho, 0, sum(item[1] for item in entropia[3]), entropia[3], implementacaoRegraFilho, termoClasse, atributo_atual.getNome()))
                            quantidadeComConfusao = 0
                countID += 1

                if(quantidadeComConfusao == 0):
                    if(not pilhaRegrasPendente):
                        controle = False
                    else:
                        indice_ultima_pendencia = next((i for i, v in enumerate(pilhaRegrasPai) if ((v[0] == pilhaRegrasPendente[-1][0]) and (v[1] == pilhaRegrasPendente[-1][1])) or ((v[0] == pilhaRegrasPendente[-1][0]) and (pilhaRegrasPendente[-1][2] == "in"))), 0)
                        del pilhaRegrasPai[indice_ultima_pendencia:]
        else:
            print('Dados de treino vazio')
    
    def testarRegistro(self, registro: pd.Series):
        noAtual = self.arvore.get_node(0)
        controle = True
        while(controle):
            if noAtual.is_leaf():
                controle = False
            else:
                identificadorAtual = noAtual.identifier
                noAtual = next((filho for filho in self.arvore.children(identificadorAtual) if filho.data.testImplementacaoRegra(registro)), None)
                if noAtual is None:
                    noAtual = max(self.arvore.children(identificadorAtual), key=lambda item: item.data.getRegistros())
        return noAtual.data.getClasse()
    
    def exibirRegras(self, idNoAtual = 0, string: str = ""):
        noAtual = self.arvore.get_node(idNoAtual)
        identificadorAtual = noAtual.identifier
        if(noAtual.is_leaf()):
            print(string + f" => Classe: {noAtual.data.getClasse()}")
            return
        else:
            for filho in self.arvore.children(identificadorAtual):
                self.exibirRegras(filho.identifier, string + (" && " if identificadorAtual != 0 else "") + filho.data.getNomeRegra())
    
    def gerar_metricas(self, x_teste: pd.DataFrame, y_teste: pd.DataFrame):
        colunasMatriz = sorted(y_teste[y_teste.columns[0]].unique())
        matrizConfusao = pd.DataFrame(
            data=np.zeros((len(colunasMatriz), len(colunasMatriz)), dtype=int),
            index=colunasMatriz,
            columns=colunasMatriz
        )
        for i in range(len(y_teste)):
            registro = x_teste.iloc[i]
            classeReal = y_teste[y_teste.columns[0]].iloc[i]
            classePrevista = self.testarRegistro(registro)
            matrizConfusao.loc[classeReal, classePrevista] += 1

        for classe in colunasMatriz:
            verdadeirosPositivos = matrizConfusao.loc[classe, classe]
            falsosPositivos = matrizConfusao[classe].sum() - verdadeirosPositivos
            falsosNegativos = matrizConfusao.loc[classe].sum() - verdadeirosPositivos
            verdadeirosNegativos = matrizConfusao.values.sum() - (verdadeirosPositivos + falsosPositivos + falsosNegativos)
            precisao = verdadeirosPositivos / (verdadeirosPositivos + falsosPositivos) if (verdadeirosPositivos + falsosPositivos) > 0 else 0
            recall = verdadeirosPositivos / (verdadeirosPositivos + falsosNegativos) if (verdadeirosPositivos + falsosNegativos) > 0 else 0
            f1_score = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
            accuracy = (verdadeirosPositivos + verdadeirosNegativos) / (verdadeirosPositivos + falsosPositivos + verdadeirosNegativos + falsosNegativos) if (verdadeirosPositivos + falsosPositivos + verdadeirosNegativos + falsosNegativos) > 0 else 0
            print(f"Classe: {classe}")
            print(f"  Precisão: {precisao:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1_score:.4f}")
            print(f"  Acurácia: {accuracy:.4f}\n")
        return matrizConfusao

                
    
