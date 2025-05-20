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
        self.df_booleano = None
        self.df_similaridade = None  # Corrigido nome do atributo
        self.idf_consulta_matrix = None

    def _gerar_vocabulario(self):
        termos = ' '.join(self.documentos).split()
        return sorted(set(termos))

    def gerar_matriz_frequencia(self):
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
        if self.tfidf_matrix is None:
            raise ValueError("A matriz TF-IDF deve ser calculada primeiro.")

        soma_colunas = self.tfidf_matrix.sum(axis=0)
        raiz_colunas = soma_colunas.pow(0.5)
        colunas_formatadas = [f'DOC {i+1}' for i in range(len(raiz_colunas))]
        df_normalizacao = pd.DataFrame([raiz_colunas.values], columns=colunas_formatadas, index=["NORMALIZAÇÃO"])
        self.normalizacao_por_soma = df_normalizacao

        print("\n📏 NORMALIZAÇÃO (raiz da soma das colunas):")
        print(df_normalizacao)

    def modelo_booleano(self, consultas):
        """
        Gera um DataFrame binário indicando se cada documento satisfaz (1) ou não (0) cada consulta.
        """
        resultados = []

        for consulta in consultas:
            termos = set(consulta.split())
            linha = []
            for doc in self.documentos:
                palavras_doc = set(doc.split())
                linha.append(1 if termos.issubset(palavras_doc) else 0)
            resultados.append(linha)

        colunas = [f'DOC {i+1}' for i in range(len(self.documentos))]
        index = [f'Consulta {i+1} ({c})' for i, c in enumerate(consultas)]

        self.df_booleano = pd.DataFrame(resultados, columns=colunas, index=index)

        print("\n🔎 Modelo Booleano:")
        print(self.df_booleano)
    
    def calcular_idfXconsulta(self, consultas):
        if self.idf_matrix is None:
            raise ValueError("A matriz IDF precisa ser calculada antes.")

        resultados = []

        for consulta in consultas:
            termos = consulta.split()
            freq = {}

            for termo in termos:
                freq[termo] = freq.get(termo, 0) + 1

            # Garante que todos os termos do vocabulário existam
            for termo in self.vocabulario:
                if termo not in freq:
                    freq[termo] = 0

            valores = []
            for termo in self.vocabulario:
                if freq[termo] > 0:
                    valor = (1 + math.log2(freq[termo])) * self.idf_matrix.loc[termo, "IDF"]
                else:
                    valor = 0
                valores.append(valor)
        
        # Calcular normalização e adicionar
            normalizacao = math.sqrt(sum([v**2 for v in valores]))
            valores.append(normalizacao)
                
            resultados.append(valores)

        # Cria DataFrame com index igual ao vocabulário e colunas C1, C2, ...
        df = pd.DataFrame(
            list(zip(*resultados)),  # transposta para termos nas linhas
            index=self.vocabulario + ['Normalização'],
            columns=[f'C{idx+1}' for idx in range(len(consultas))]
        )

        self.idf_consulta_matrix = df

        print("\n📘 Matriz IDF × Consulta:")
        print(self.idf_consulta_matrix)


    def calcular_similaridade_vetorial(self, consultas):
        """
        Gera um DataFrame com as similaridades de cosseno entre as consultas e os documentos.
        """
        if self.tfidf_matrix is None:
            raise ValueError("A matriz TF-IDF precisa ser calculada.")

        resultados = []

        for consulta in consultas:
            termos_consulta = consulta.split()
            vetor_consulta = []

            for termo in self.vocabulario:
                freq = termos_consulta.count(termo)
                if freq > 0:
                    tf = 1 + math.log2(freq)
                    idf = self.idf_matrix.loc[termo, "IDF"]
                    vetor_consulta.append(tf * idf)
                else:
                    vetor_consulta.append(0.0)

            linha_similaridades = []
            for i in range(len(self.documentos)):
                vetor_doc = self.tfidf_matrix[f'D{i+1}'].values.tolist()
                produto_escalar = sum(c * d for c, d in zip(vetor_consulta, vetor_doc))
                norma_consulta = math.sqrt(sum(c ** 2 for c in vetor_consulta))
                norma_doc = math.sqrt(sum(d ** 2 for d in vetor_doc))
                similaridade = produto_escalar / (norma_consulta * norma_doc) if norma_consulta and norma_doc else 0.0
                linha_similaridades.append(round(similaridade, 4))

            resultados.append(linha_similaridades)

        colunas = [f'DOC {i+1}' for i in range(len(self.documentos))]
        index = [f'Consulta {i+1} ({c})' for i, c in enumerate(consultas)]

        self.df_similaridade = pd.DataFrame(resultados, columns=colunas, index=index)

        print("\n📐 Similaridade Vetorial:")
        print(self.df_similaridade)
    
    def exportFile(self, nome_arquivo):
        with pd.ExcelWriter(nome_arquivo, engine='xlsxwriter') as writer:
            # Cria planilha vazia para pegar worksheet
            pd.DataFrame().to_excel(writer, sheet_name='Planilha1')
            ws = writer.sheets['Planilha1']

            coluna_atual = 0
            linha_titulo = 0  # título fica na linha 0, os dados começam na linha 1

            def escrever_titulo(texto, linha, coluna):
                ws.write(linha, coluna, texto)

            def salvar_df_com_titulo(df, titulo, linha_inicio, coluna_inicio):
                escrever_titulo(titulo, linha_inicio, coluna_inicio)
                df.to_excel(writer, sheet_name='Planilha1', startrow=linha_inicio + 1, startcol=coluna_inicio, header=True, index=True)
                return coluna_inicio + df.shape[1] + 3  # pula 3 colunas para próxima tabela

            if self.matriz_frequencia is not None:
                coluna_atual = salvar_df_com_titulo(self.matriz_frequencia, "Matriz Frequência", linha_titulo, coluna_atual)

            if self.tf_matrix is not None:
                coluna_atual = salvar_df_com_titulo(self.tf_matrix, "Matriz TF", linha_titulo, coluna_atual)

            if self.idf_matrix is not None:
                coluna_atual = salvar_df_com_titulo(self.idf_matrix, "Matriz IDF", linha_titulo, coluna_atual)

            if self.tfidf_matrix is not None:
                coluna_atual = salvar_df_com_titulo(self.tfidf_matrix, "Matriz TF-IDF", linha_titulo, coluna_atual)

            if hasattr(self, 'normalizacao_por_soma'):
                coluna_atual = salvar_df_com_titulo(self.normalizacao_por_soma, "Normalização TF-IDF", linha_titulo, coluna_atual)

            if self.df_booleano is not None:
                coluna_atual = salvar_df_com_titulo(self.df_booleano, "Modelo Booleano (binário)", linha_titulo, coluna_atual)

            if hasattr(self, 'idf_consulta_matrix'):
                coluna_atual = salvar_df_com_titulo(self.idf_consulta_matrix, "1 log(freq) × idf", linha_titulo, coluna_atual)

            if hasattr(self, 'df_similaridade'):
                coluna_atual = salvar_df_com_titulo(self.df_similaridade, "Similaridade Vetorial (coseno)", linha_titulo, coluna_atual)







# --- EXEMPLO DE USO ---
documentos = [
    "homem estar tempo coisa dizer ir ter",
    "senhora estar dia moço moço senhora",
    "senhora vez senhora senhora tempo dizer filho",
    "casa ir ir dizer ter olho",
    "olho dia vez dia homem moço tempo",
]

consultas = [
    "homem moço",
    "dizer ir tempo",
    "dia senhora casa",
]

matriz_tfidf = MatrizTFIDF(documentos)
matriz_tfidf.gerar_matriz_frequencia()
matriz_tfidf.calcular_tf()
matriz_tfidf.calcular_idf()
matriz_tfidf.calcular_tfidf()
matriz_tfidf.normalizar_tfidf_por_soma()
matriz_tfidf.modelo_booleano(consultas)
matriz_tfidf.calcular_idfXconsulta(consultas)
matriz_tfidf.calcular_similaridade_vetorial(consultas)
matriz_tfidf.exportFile("atv2.xlsx")


