import pandas as pd
import treelib as tl
import math
import numpy as np

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
        return f"Regra: {self.nomeRegra} \n Atributo: {self.atributo} \n Entropia: {self.entropia} \n Registros: {self.registros} \n ClassesPrev: {self.classesPrev} \n ImplementacaoRegra: {self.implementacaoRegra}"
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
    def printAll(self):
        self.print()
        print(f"Ganho: {self.ganho}")
        print(f"ValoresAtributos: {self.valoresAtributos}")
    def __str__(self):
        return f"Nome: {self.Nome} \n Entropia: {self.entropia} \n Registros: {self.registros} \n ClassesPrev: {self.classesPrev} \n Ganho: {self.ganho} \n ValoresAtributos: {self.valoresAtributos}"
    def __repr__(self):
        return self.__str__()


class ArvoreID3:
    def __init__(self):
        self.arvore = tl.Tree()

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

    def __calcular_entropiaClasse(self, coluna : pd.DataFrame):
        distribuicaoValoresDiferentes = coluna.value_counts()
        quantidadeRegistros = len(coluna)
        entropia = 0
        for valor, quantidade in distribuicaoValoresDiferentes.items():
            entropia  += -(quantidade/quantidadeRegistros) * math.log((quantidade/quantidadeRegistros), 2)

        return entropia
    def __calcular_entropiaAtributo(self, atributo: pd.DataFrame, classe: pd.DataFrame):
        distribuicaoValoresDiferentesAtrib = atributo.value_counts()

        concatenacaoAtribClass = pd.concat([atributo, classe], axis=1)
        concatenacaoAtribClass = concatenacaoAtribClass.dropna(axis=0, how='any', subset=[atributo.columns[0]])
        distribuicaoValoresDiferentesClasse = concatenacaoAtribClass[classe.columns[0]].value_counts()
        quantidadeRegistros = len(atributo)
        entropia = 0
        testeAtributo = 0
        quantidadeClasseporAtributo = 0
        listaEntropías = list()
        quantidadeResultadoClasse = list()
        for valorAtributo, quantidadeAtributo in distribuicaoValoresDiferentesAtrib.items():
            for valorClasse, quantidadeClasse in distribuicaoValoresDiferentesClasse.items():
                quantidadeClasseporAtributo = len(concatenacaoAtribClass[(concatenacaoAtribClass[atributo.columns[0]] == valorAtributo[0]) & (concatenacaoAtribClass[classe.columns[0]] == valorClasse)])
                quantidadeResultadoClasse.append((valorClasse,quantidadeClasseporAtributo))
                if(quantidadeClasseporAtributo > 0):
                    testeAtributo += -(quantidadeClasseporAtributo/quantidadeAtributo) * math.log((quantidadeClasseporAtributo/quantidadeAtributo), 2)

            entropia += (quantidadeAtributo/quantidadeRegistros)*testeAtributo
            listaEntropías.append((valorAtributo[0], testeAtributo, quantidadeResultadoClasse.copy()))
            testeAtributo = 0
            quantidadeResultadoClasse.clear()
        return entropia, list(distribuicaoValoresDiferentesClasse.items()), listaEntropías
    

    def __atributo_maior_ganho(self, atributos: pd.DataFrame, classe: pd.DataFrame, filtros: list = list()):
        ganhoAtributos = list()
        mascara = pd.Series([True] * len(atributos))
        for filtro in filtros:
            mascara = mascara & (atributos[filtro[0]] == filtro[1])
        registrosFiltrados = atributos[mascara]
        entropiaClasse = self.__calcular_entropiaClasse(classe[mascara])
        for coluna in registrosFiltrados.columns:
            if not any(filtro[0] == coluna for filtro in filtros):
                entropiaAtributo, registrosClasses, listaEntropias = self.__calcular_entropiaAtributo(registrosFiltrados[[coluna]], classe)
                ganhoAtributos.append(Categoria(coluna, entropiaAtributo, (entropiaClasse - entropiaAtributo), sum(item[1] for item in registrosClasses), registrosClasses, listaEntropias))
        return max(ganhoAtributos, key=lambda item: item.getGanho())
    
    
    
    def __create_node(self, tag, identifier, parent:int | None = None, data: No | None = None):
        self.arvore.create_node(tag=tag, identifier=identifier, parent=parent, data=data)
    def show(self):
        self.arvore.show()

    def gerar_arvore_ID3(self, x_treino: pd.DataFrame, y_treino: pd.DataFrame):
        if(len(x_treino) > 0 and len(y_treino) > 0):    
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
                    regrasPaiImplementacao = lambda registo, coluna = pilhaRegrasPai[-1][0], valor = pilhaRegrasPai[-1][1]: registo[coluna] == valor
                    strRegrasPai = f"{pilhaRegrasPai[-1][0]} == {pilhaRegrasPai[-1][1]}"
                else:
                    regrasPaiImplementacao = lambda registo: True
                    strRegrasPai = "True"

                atributo_atual = self.__atributo_maior_ganho(baseXTreino, baseYTreino, pilhaRegrasPai)
                No_atual = No(strRegrasPai, atributo_atual.entropia, atributo_atual.registros, atributo_atual.classesPrev, regrasPaiImplementacao, None, atributo_atual.getNome())
                
                self.__create_node(tag=f"{strRegrasPai} -> Testar: {atributo_atual.getNome()} Qtd: {atributo_atual.getRegistros()}", identifier=countID, parent= None if countID==0 else pilhaRegrasPai[-1][2], data=No_atual)
                quantidadeComConfusao = sum(1 for ent in atributo_atual.getValoresAtributos() if ent[1] > 0)

                paiID = countID
                dentroDoLimiteDeRegras = (len(pilhaRegrasPai) + 1) < x_treino.shape[1]
                if(dentroDoLimiteDeRegras):
                    for entropia in atributo_atual.getValoresAtributos():
                        if entropia[1] > 0:
                            pilhaRegrasPendente.append((atributo_atual.getNome(), entropia[0], paiID))
                        else:
                            implementacaoRegraFilho = lambda registro, coluna=atributo_atual.getNome(), valor=entropia[0]: registro[coluna] == valor
                            strRegrasFilho = f"{atributo_atual.getNome()} == {entropia[0]}"
                            countID += 1
                            termoClasse = max(entropia[2], key= lambda item: item[1] )[0]
                            self.__create_node(tag=f"{strRegrasFilho} -> {termoClasse} Qtd: {sum(item[1] for item in entropia[2])}", identifier=countID, parent=paiID, data=
                                            No(strRegrasFilho, 0, sum(item[1] for item in entropia[2]), entropia[2], implementacaoRegraFilho, termoClasse, atributo_atual.getNome()))
                else:
                    for entropia in atributo_atual.getValoresAtributos():
                        implementacaoRegraFilho = lambda registro, coluna=atributo_atual.getNome(), valor=entropia[0]: registro[coluna] == valor
                        strRegrasFilho = f"{atributo_atual.getNome()} == {entropia[0]}"
                        countID += 1
                        termoClasse = max(entropia[2], key= lambda item: item[1] )[0]
                        self.__create_node(tag=f"{strRegrasFilho} -> {termoClasse} Qtd: {sum(item[1] for item in entropia[2])}", identifier=countID, parent=paiID, data=
                                        No(strRegrasFilho, 0, sum(item[1] for item in entropia[2]), entropia[2], implementacaoRegraFilho, termoClasse, atributo_atual.getNome()))
                countID = countID + 1

                if(quantidadeComConfusao == 0 or not dentroDoLimiteDeRegras):
                    if(not pilhaRegrasPendente):
                        controle = False
                    else:
                        indice_ultima_pendencia = next((i for i, v in enumerate(pilhaRegrasPai) if v[0] == pilhaRegrasPendente[-1][0]), None)
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
