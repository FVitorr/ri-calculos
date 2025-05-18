import math
import pandas as pd

class MatrizTFIDF:
    def __init__(self, documentos):
        self.documentos = documentos
        self.vocabulario = self._gerar_vocabulario()
        self.matriz_frequencia = None
        self.tf_matrix = None
        self.idf_matrix = None
        self.tfidf_matrix = None

    def _gerar_vocabulario(self):
        termos = ' '.join(self.documentos).split()
        return sorted(set(termos))

    def gerar_matriz_frequencia(self):
        """
        Gera a matriz de frequência (frequência absoluta de termos).
        """
        dados = []
        for termo in self.vocabulario:
            dados.append([doc.split().count(termo) for doc in self.documentos])

        self.matriz_frequencia = pd.DataFrame(
            dados,
            columns=[f'D{idx+1}' for idx in range(len(self.documentos))],
            index=self.vocabulario
        )
        print("📊 Matriz de Frequência (Frequência Absoluta):")
        print(self.matriz_frequencia)

    def calcular_tf(self):
        """
        Calcula a matriz TF com base na frequência absoluta.
        TF(i,j) = 1 + log2(f(i,j)) se f(i,j) > 0, senão 0
        """
        if self.matriz_frequencia is None:
            raise ValueError("A matriz de frequência deve ser gerada primeiro.")

        tf_values = []
        for termo in self.vocabulario:
            tf_linha = [
                1 + math.log2(self.matriz_frequencia.at[termo, f'D{idx+1}']) if self.matriz_frequencia.at[termo, f'D{idx+1}'] > 0 else 0
                for idx in range(len(self.documentos))
            ]
            tf_values.append(tf_linha)

        self.tf_matrix = pd.DataFrame(
            tf_values,
            columns=[f'D{idx+1}' for idx in range(len(self.documentos))],
            index=self.vocabulario
        )
        print("\n📘 Matriz TF:")
        print(self.tf_matrix)

    def calcular_idf(self):
        """
        Calcula a IDF para cada termo usando log2(N / ni).
        """
        if self.matriz_frequencia is None:
            raise ValueError("A matriz de frequência deve ser gerada primeiro.")
        
        N = len(self.documentos)
        idf_values = {}
        for termo in self.vocabulario:
            ni = (self.matriz_frequencia.loc[termo] > 0).sum()
            idf_values[termo] = math.log2(N / ni) if ni > 0 else 0

        self.idf_matrix = pd.DataFrame(
            list(idf_values.items()), columns=["Termo", "IDF"]
        ).set_index("Termo")
        print("\n📗 Matriz IDF:")
        print(self.idf_matrix)

    def calcular_tfidf(self):
        """
        Calcula a matriz TF-IDF multiplicando TF pelo IDF.
        """
        if self.tf_matrix is None or self.idf_matrix is None:
            raise ValueError("As matrizes TF e IDF devem ser calculadas primeiro.")

        tfidf_values = []
        for termo in self.vocabulario:
            linha_tfidf = self.tf_matrix.loc[termo] * self.idf_matrix.loc[termo, "IDF"]
            tfidf_values.append(linha_tfidf)

        self.tfidf_matrix = pd.DataFrame(
            tfidf_values,
            columns=[f'D{idx+1}' for idx in range(len(self.documentos))],
            index=self.vocabulario
        )
        print("\n📙 Matriz TF-IDF:")
        print(self.tfidf_matrix)
    
    def normalizar_tfidf_por_soma(self):
        """
        Calcula a raiz quadrada da soma dos valores de cada coluna (documento) da matriz TF-IDF
        e retorna em um DataFrame com índice 'NORMALIZAÇÃO'.
        """
        if self.tfidf_matrix is None:
            raise ValueError("A matriz TF-IDF deve ser calculada primeiro.")

        # Soma total de cada coluna
        soma_colunas = self.tfidf_matrix.sum(axis=0)

        # Aplica raiz quadrada
        raiz_colunas = soma_colunas.pow(0.5)

        # Renomeia colunas para o padrão "DOC 1", "DOC 2", ...
        colunas_formatadas = [f'DOC {i+1}' for i in range(len(raiz_colunas))]

        # Cria o DataFrame final com índice "NORMALIZAÇÃO"
        df_normalizacao = pd.DataFrame([raiz_colunas.values], columns=colunas_formatadas, index=["NORMALIZAÇÃO"])

        print("\n📏 NORMALIZAÇÃO (raiz da soma das colunas):")
        print(df_normalizacao)

        # Armazena se precisar usar depois
        self.normalizacao_por_soma = df_normalizacao
    
    # Similaridade entre os documentos e cada uma das consultas utilizando o modelo booleano
    def modelo_booleano(self, consultas):
        print("\n🔍 Similaridade - Modelo Booleano (AND):")
        for idx, consulta in enumerate(consultas):
            termos = set(consulta.split())
            resultados = []
            for i, doc in enumerate(self.documentos):
                palavras_doc = set(doc.split())
                if termos.issubset(palavras_doc):
                    resultados.append(f"D{i+1}")
            print(f"Consulta {idx+1} ({consulta}): Documentos -> {resultados if resultados else 'Nenhum'}")

    def calcular_similaridade_vetorial(self, consultas):
        if self.tfidf_matrix is None:
            raise ValueError("A matriz TF-IDF precisa ser calculada.")

        print("\n📐 Similaridade - Modelo Vetorial (Cosseno - sem sklearn):")

        for idx, consulta in enumerate(consultas):
            termos_consulta = consulta.split()
            vetor_consulta = []

            # Monta vetor TF-IDF da consulta
            for termo in self.vocabulario:
                freq = termos_consulta.count(termo)
                if freq > 0:
                    tf = 1 + math.log2(freq)
                    idf = self.idf_matrix.loc[termo, "IDF"]
                    vetor_consulta.append(tf * idf)
                else:
                    vetor_consulta.append(0.0)

            # Similaridade com cada documento
            for i in range(len(self.documentos)):
                vetor_doc = self.tfidf_matrix[f'D{i+1}'].values.tolist()

                # Produto escalar
                produto_escalar = sum(c * d for c, d in zip(vetor_consulta, vetor_doc))

                # Normas
                norma_consulta = math.sqrt(sum(c ** 2 for c in vetor_consulta))
                norma_doc = math.sqrt(sum(d ** 2 for d in vetor_doc))

                # Similaridade de cosseno
                if norma_consulta == 0 or norma_doc == 0:
                    similaridade = 0.0
                else:
                    similaridade = produto_escalar / (norma_consulta * norma_doc)

                print(f"Consulta {idx+1} ({consulta}) vs Documento D{i+1}: Similaridade = {similaridade:.4f}")

       

# --- EXEMPLO DE USO ---
documentos = [
    "nota prova avaliação sala data prova",
    "aluno nota avaliação sala nota",
    "aluno prova avaliação",
]

consulta = [
    "nota aluno"
    "data prova"
]

matriz_tfidf = MatrizTFIDF(documentos)
matriz_tfidf.gerar_matriz_frequencia()
matriz_tfidf.calcular_tf()
matriz_tfidf.calcular_idf()
matriz_tfidf.calcular_tfidf()
matriz_tfidf.normalizar_tfidf_por_soma()

