"""
knn.py - k-vizinhos mais proximos implementado do zero.

Cobre a Secao 4.4 do PP01:
  1. voto majoritario e voto ponderado pelo inverso da distancia;
  2. euclidiana, Manhattan, Chebyshev, Minkowski(p), cosseno e HEOM (dados mistos);
  3. atributos heterogeneos: HEOM compara categoricos por igualdade (0/1) e
     numericos por diferenca normalizada pela faixa;
  4. varredura de k reaproveitando a mesma matriz de distancias.

Dependencia: apenas numpy. Nenhum uso de KNeighborsClassifier.
"""

from __future__ import annotations

import numpy as np

METRICAS = ("euclidiana", "manhattan", "chebyshev", "minkowski", "cosseno", "heom")
_EPS = 1e-12


# ---------------------------------------------------------------------------
# 1. Distancia entre dois vetores (versao didatica, para o relatorio)
# ---------------------------------------------------------------------------


def distancia(a, b, metrica: str = "euclidiana", p: float = 2.0,
              cat: np.ndarray | None = None, faixas: np.ndarray | None = None) -> float:
    """Distancia entre dois pontos.

    euclidiana : sqrt(sum (a_i - b_i)^2)          -> Minkowski com p=2
    manhattan  : sum |a_i - b_i|                  -> Minkowski com p=1
    chebyshev  : max |a_i - b_i|                  -> Minkowski com p->inf
    minkowski  : (sum |a_i - b_i|^p)^(1/p)
    cosseno    : 1 - (a.b)/(|a||b|)               -> mede angulo, ignora magnitude
    heom       : mistura: 0/1 nos categoricos, |a-b|/faixa nos numericos
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    dif = np.abs(a - b)
    if metrica == "euclidiana":
        return float(np.sqrt((dif ** 2).sum()))
    if metrica == "manhattan":
        return float(dif.sum())
    if metrica == "chebyshev":
        return float(dif.max()) if dif.size else 0.0
    if metrica == "minkowski":
        return float((dif ** p).sum() ** (1.0 / p))
    if metrica == "cosseno":
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(1.0 - a @ b / (na * nb + _EPS))
    if metrica == "heom":
        if cat is None:
            cat = np.zeros(a.size, dtype=bool)
        if faixas is None:
            faixas = np.ones(a.size)
        d = np.where(cat, (a != b).astype(float), dif / np.maximum(faixas, _EPS))
        return float(np.sqrt((d ** 2).sum()))
    raise ValueError(f"metrica desconhecida: {metrica}")


# ---------------------------------------------------------------------------
# 2. Matriz de distancias (vetorizada, em blocos para nao estourar a memoria)
# ---------------------------------------------------------------------------


def _dist_bloco(A: np.ndarray, B: np.ndarray, metrica: str, p: float,
                cat: np.ndarray | None, faixas: np.ndarray | None) -> np.ndarray:
    """Distancias entre todas as linhas de A e todas as linhas de B -> (|A|, |B|)."""
    if metrica == "euclidiana":
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (evita o tensor 3D)
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
        return np.sqrt(np.maximum(d2, 0.0))
    if metrica == "cosseno":
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + _EPS)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + _EPS)
        return 1.0 - An @ Bn.T
    if metrica == "heom":
        dif = A[:, None, :] - B[None, :, :]
        d = np.where(cat[None, None, :], (dif != 0).astype(float),
                     np.abs(dif) / np.maximum(faixas, _EPS)[None, None, :])
        return np.sqrt((d ** 2).sum(axis=2))

    dif = np.abs(A[:, None, :] - B[None, :, :])
    if metrica == "manhattan":
        return dif.sum(axis=2)
    if metrica == "chebyshev":
        return dif.max(axis=2)
    if metrica == "minkowski":
        return (dif ** p).sum(axis=2) ** (1.0 / p)
    raise ValueError(f"metrica desconhecida: {metrica}")


# ---------------------------------------------------------------------------
# 3. Classificador
# ---------------------------------------------------------------------------


class KNN:
    """k-NN por forca bruta (aprendizado preguicoso: o `fit` so memoriza).

    Parametros
    ----------
    k          : numero de vizinhos
    metrica    : ver METRICAS
    p          : expoente da Minkowski
    ponderado  : False = voto majoritario; True = voto ponderado por 1/d
    colunas_categoricas : indices booleanos/inteiros das colunas categoricas (so p/ HEOM)
    max_elem   : orcamento de memoria por bloco (numero de elementos do tensor)
    """

    def __init__(self, k: int = 5, metrica: str = "euclidiana", p: float = 3.0,
                 ponderado: bool = False, colunas_categoricas=None,
                 max_elem: int = 4_000_000):
        if metrica not in METRICAS:
            raise ValueError(f"metrica deve ser uma de {METRICAS}")
        self.k = int(k)
        self.metrica = metrica
        self.p = float(p)
        self.ponderado = bool(ponderado)
        self.colunas_categoricas = colunas_categoricas
        self.max_elem = int(max_elem)

    # ------------------------------------------------------------------ fit
    def fit(self, X, y) -> "KNN":
        """Aprendizado preguicoso: apenas guarda o conjunto de treino."""
        self.X_ = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        d = self.X_.shape[1]

        self._cat = np.zeros(d, dtype=bool)
        if self.colunas_categoricas is not None:
            self._cat[np.asarray(self.colunas_categoricas)] = True
        # faixa (max-min) de cada atributo, estimada SO no treino
        self._faixas = self.X_.max(axis=0) - self.X_.min(axis=0)
        self._faixas[self._faixas == 0] = 1.0
        return self

    # ------------------------------------------------------- distancias
    def matriz_distancias(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_t, d = self.X_.shape
        precisa_3d = self.metrica in ("manhattan", "chebyshev", "minkowski", "heom")
        passo = max(1, self.max_elem // max(n_t * d, 1)) if precisa_3d else 1024
        blocos = [
            _dist_bloco(X[i:i + passo], self.X_, self.metrica, self.p, self._cat, self._faixas)
            for i in range(0, X.shape[0], passo)
        ]
        return np.vstack(blocos)

    # ------------------------------------------------------------- votacao
    def _votar(self, D: np.ndarray, k: int) -> np.ndarray:
        """Recebe a matriz de distancias e devolve as probabilidades por classe."""
        k = min(k, D.shape[1])
        viz = np.argpartition(D, k - 1, axis=1)[:, :k]          # k menores (sem ordenar)
        d_viz = np.take_along_axis(D, viz, axis=1)
        rot = self.y_[viz]                                       # (n, k)

        if self.ponderado:
            pesos = 1.0 / (d_viz + _EPS)
            # se algum vizinho esta a distancia zero, ele domina o voto
            exato = d_viz <= _EPS
            tem_exato = exato.any(axis=1)
            pesos[tem_exato] = exato[tem_exato].astype(float)
        else:
            pesos = np.ones_like(d_viz)

        votos = np.zeros((D.shape[0], len(self.classes_)))
        linhas = np.repeat(np.arange(D.shape[0]), k)
        np.add.at(votos, (linhas, rot.ravel()), pesos.ravel())
        soma = votos.sum(axis=1, keepdims=True)
        return votos / np.maximum(soma, _EPS)

    # ------------------------------------------------------------ predicao
    def predict_proba(self, X) -> np.ndarray:
        return self._votar(self.matriz_distancias(X), self.k)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def prever_varios_k(self, X, lista_k) -> dict:
        """Curva de k sem recalcular distancias: a matriz D e reaproveitada.

        Devolve {k: vetor de predicoes}. Usar na varredura k in {1,3,...,51}.
        """
        D = self.matriz_distancias(X)
        return {int(k): self.classes_[np.argmax(self._votar(D, int(k)), axis=1)]
                for k in lista_k}


def classificar_knn(X_treino, y_treino, x, k: int = 5, metrica: str = "euclidiana",
                    ponderado: bool = False, p: float = 3.0):
    """Assinatura pedida no enunciado: classifica um unico ponto x."""
    modelo = KNN(k=k, metrica=metrica, p=p, ponderado=ponderado).fit(X_treino, y_treino)
    return modelo.predict(np.asarray(x, dtype=float).reshape(1, -1))[0]


# ---------------------------------------------------------------------------
# 4. Normalizadores (o k-NN depende deles; a arvore, nao)
# ---------------------------------------------------------------------------


class MinMax:
    """Escalona para [0, 1]. Ajustar SO no treino."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        faixa = X.max(axis=0) - self.min_
        faixa[faixa == 0] = 1.0
        self.faixa_ = faixa
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.min_) / self.faixa_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class ZScore:
    """Padronizacao (media 0, desvio 1). Ajustar SO no treino."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.media_ = X.mean(axis=0)
        dp = X.std(axis=0)
        dp[dp == 0] = 1.0
        self.dp_ = dp
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.media_) / self.dp_

    def fit_transform(self, X):
        return self.fit(X).transform(X)