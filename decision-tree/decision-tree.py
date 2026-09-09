"""
arvore.py - Arvore de decisao (ID3/CART simplificada) implementada do zero.

Cobre a Secao 4.3 do PP01:
  1. entropia, ganho de informacao e indice Gini calculados na mao (sem sklearn);
  2. inducao com divisao binaria por limiar (v <= t) e parada por profundidade
     maxima / numero minimo de amostras;
  3. extracao de regras SE-ENTAO com cobertura e confianca;
  4. poda por erro reduzido (reduced error pruning) sobre um conjunto de validacao.

Dependencia: apenas numpy (algebra/vetorizacao, nao e biblioteca de ML).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 1. Medidas de impureza
# ---------------------------------------------------------------------------


def entropia(y) -> float:
    """Entropia de Shannon (base 2) do vetor de rotulos y.

    H(y) = - sum_c p_c * log2(p_c),  com p_c = n_c / n.
    Vale 0 quando o no e puro e log2(K) quando as K classes sao equiprovaveis.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, cont = np.unique(y, return_counts=True)
    p = cont / y.size
    return float(-(p * np.log2(p)).sum())


def gini(y) -> float:
    """Indice de Gini: 1 - sum_c p_c^2. Probabilidade de errar rotulando ao acaso."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, cont = np.unique(y, return_counts=True)
    p = cont / y.size
    return float(1.0 - (p ** 2).sum())


def _impureza(y, criterio: str) -> float:
    return entropia(y) if criterio == "entropia" else gini(y)


def _impureza_lote(cont: np.ndarray, n: np.ndarray, criterio: str) -> np.ndarray:
    """Impureza de varios nos de uma vez.

    cont : (m, K) contagens por classe de m candidatos a no
    n    : (m,)   total de amostras de cada candidato
    Usado na busca vetorizada de limiares; matematicamente identico as funcoes acima.
    """
    p = cont / n[:, None]
    if criterio == "gini":
        return 1.0 - (p ** 2).sum(axis=1)
    log = np.zeros_like(p)
    np.log2(p, out=log, where=p > 0)
    return -(p * log).sum(axis=1)


def ganho_informacao(coluna, y, limiar: float, criterio: str = "entropia") -> float:
    """Ganho da divisao binaria `coluna <= limiar`.

    G = I(pai) - (n_esq/n) * I(esq) - (n_dir/n) * I(dir)

    Com criterio='gini' o valor retornado e a "diminuicao de impureza de Gini",
    que e o criterio usado pelo CART / DecisionTreeClassifier(criterion='gini').
    """
    coluna = np.asarray(coluna, dtype=float)
    y = np.asarray(y)
    mask = coluna <= limiar
    n = y.size
    n_e = int(mask.sum())
    n_d = n - n_e
    if n_e == 0 or n_d == 0:
        return 0.0
    return float(
        _impureza(y, criterio)
        - (n_e / n) * _impureza(y[mask], criterio)
        - (n_d / n) * _impureza(y[~mask], criterio)
    )


def razao_de_ganho(coluna, y, limiar: float, criterio: str = "entropia") -> float:
    """Gain ratio (C4.5): ganho normalizado pela entropia da propria divisao.

    Penaliza divisoes que quebram o no em pedacos muito desiguais.
    """
    coluna = np.asarray(coluna, dtype=float)
    mask = coluna <= limiar
    n = mask.size
    n_e = int(mask.sum())
    if n_e == 0 or n_e == n:
        return 0.0
    p = np.array([n_e / n, 1 - n_e / n])
    split_info = float(-(p * np.log2(p)).sum())
    return ganho_informacao(coluna, y, limiar, criterio) / split_info


# ---------------------------------------------------------------------------
# 2. Estrutura da arvore
# ---------------------------------------------------------------------------


class No:
    """No da arvore. E folha quando `atributo is None`."""

    __slots__ = ("atributo", "limiar", "esq", "dir", "prob", "classe", "n", "imp", "ganho")

    def __init__(self, prob: np.ndarray, n: int, imp: float):
        self.atributo: int | None = None
        self.limiar: float | None = None
        self.esq: "No | None" = None
        self.dir: "No | None" = None
        self.prob = prob        # distribuicao de classes do no (vetor de tamanho K)
        self.classe = int(np.argmax(prob))
        self.n = n              # amostras de treino que caem no no
        self.imp = imp          # impureza do no
        self.ganho = 0.0        # ganho obtido na divisao (0 nas folhas)

    @property
    def folha(self) -> bool:
        return self.atributo is None

    def virar_folha(self) -> None:
        self.atributo = None
        self.limiar = None
        self.esq = None
        self.dir = None
        self.ganho = 0.0


class ArvoreDecisao:
    """ID3/CART simplificada: divisao binaria por limiar, criterio entropia ou gini.

    Parametros
    ----------
    criterio   : 'entropia' | 'gini' | 'razao_ganho'
    prof_max   : profundidade maxima (None = sem limite)
    min_folha  : numero minimo de amostras em cada folha
    min_divisao: numero minimo de amostras para tentar dividir um no
    ganho_min  : ganho minimo aceitavel para efetuar a divisao (pre-poda)
    """

    def __init__(
        self,
        criterio: str = "entropia",
        prof_max: int | None = None,
        min_folha: int = 1,
        min_divisao: int = 2,
        ganho_min: float = 0.0,
    ):
        if criterio not in ("entropia", "gini", "razao_ganho"):
            raise ValueError("criterio deve ser 'entropia', 'gini' ou 'razao_ganho'")
        self.criterio = criterio
        self.prof_max = prof_max
        self.min_folha = max(1, int(min_folha))
        self.min_divisao = max(2, int(min_divisao))
        self.ganho_min = float(ganho_min)
        self.raiz: No | None = None
        self.classes_: np.ndarray | None = None

    # -------------------------------------------------- utilitarios internos
    def _codificar_y(self, y) -> np.ndarray:
        """Mapeia os rotulos originais para indices 0..K-1."""
        y = np.asarray(y)
        idx = np.searchsorted(self.classes_, y)
        return idx.astype(np.int64)

    def _melhor_divisao(self, X: np.ndarray, y: np.ndarray):
        """Procura (atributo, limiar) de maior ganho.

        Para cada coluna: ordena os valores uma vez e varre os pontos de corte
        acumulando as contagens por classe (soma prefixa). Isso avalia todos os
        limiares candidatos em O(n log n) por atributo, em vez de recomputar a
        impureza do zero para cada candidato.
        """
        n, d = X.shape
        K = len(self.classes_)
        cont_pai = np.bincount(y, minlength=K).astype(float)
        imp_pai = float(_impureza_lote(cont_pai[None, :], np.array([n], float), self.criterio)[0])

        melhor_ganho, melhor_j, melhor_t = self.ganho_min, None, None

        for j in range(d):
            col = X[:, j]
            ordem = np.argsort(col, kind="mergesort")
            cv, yv = col[ordem], y[ordem]

            # posicoes i onde cv[i] < cv[i+1] -> corte valido entre i e i+1
            cortes = np.flatnonzero(cv[:-1] < cv[1:])
            if cortes.size == 0:
                continue
            n_esq = cortes + 1
            n_dir = n - n_esq
            ok = (n_esq >= self.min_folha) & (n_dir >= self.min_folha)
            cortes, n_esq, n_dir = cortes[ok], n_esq[ok], n_dir[ok]
            if cortes.size == 0:
                continue

            onehot = np.zeros((n, K))
            onehot[np.arange(n), yv] = 1.0
            acum = np.cumsum(onehot, axis=0)          # acum[i] = contagens de 0..i
            cont_esq = acum[cortes]                   # (m, K)
            cont_dir = cont_pai[None, :] - cont_esq

            imp_e = _impureza_lote(cont_esq, n_esq.astype(float), self.criterio)
            imp_d = _impureza_lote(cont_dir, n_dir.astype(float), self.criterio)
            ganho = imp_pai - (n_esq / n) * imp_e - (n_dir / n) * imp_d

            if self.criterio == "razao_ganho":
                pe = n_esq / n
                split_info = -(pe * np.log2(pe) + (1 - pe) * np.log2(1 - pe))
                ganho = ganho / np.maximum(split_info, 1e-12)

            b = int(np.argmax(ganho))
            if ganho[b] > melhor_ganho:
                melhor_ganho = float(ganho[b])
                melhor_j = j
                melhor_t = float((cv[cortes[b]] + cv[cortes[b] + 1]) / 2.0)

        return melhor_ganho, melhor_j, melhor_t

    def _construir(self, X: np.ndarray, y: np.ndarray, prof: int) -> No:
        K = len(self.classes_)
        cont = np.bincount(y, minlength=K).astype(float)
        prob = cont / cont.sum()
        no = No(prob, n=y.size, imp=float(_impureza_lote(cont[None, :], np.array([y.size], float), self.criterio)[0]))

        # criterios de parada
        if (
            no.imp == 0.0
            or y.size < self.min_divisao
            or (self.prof_max is not None and prof >= self.prof_max)
        ):
            return no

        ganho, j, t = self._melhor_divisao(X, y)
        if j is None:
            return no

        mask = X[:, j] <= t
        no.atributo, no.limiar, no.ganho = j, t, ganho
        no.esq = self._construir(X[mask], y[mask], prof + 1)
        no.dir = self._construir(X[~mask], y[~mask], prof + 1)
        return no

    # ------------------------------------------------------------ interface
    def fit(self, X, y) -> "ArvoreDecisao":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X deve ser 2D")
        self.classes_ = np.unique(y)
        self.raiz = self._construir(X, self._codificar_y(y), prof=0)
        return self

    def _descer(self, x: np.ndarray) -> No:
        no = self.raiz
        while not no.folha:
            no = no.esq if x[no.atributo] <= no.limiar else no.dir
        return no

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.vstack([self._descer(x).prob for x in X])

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.classes_[np.array([self._descer(x).classe for x in X])]

    # ------------------------------------------------------------ inspecao
    def profundidade(self, no: No | None = None) -> int:
        no = self.raiz if no is None else no
        if no.folha:
            return 0
        return 1 + max(self.profundidade(no.esq), self.profundidade(no.dir))

    def n_folhas(self, no: No | None = None) -> int:
        no = self.raiz if no is None else no
        return 1 if no.folha else self.n_folhas(no.esq) + self.n_folhas(no.dir)

    def importancia_atributos(self, d: int) -> np.ndarray:
        """Importancia = soma das reducoes de impureza ponderadas por n, normalizada.

        Mesma definicao do `feature_importances_` do scikit-learn.
        """
        imp = np.zeros(d)
        n_raiz = self.raiz.n

        def caminhar(no: No):
            if no.folha:
                return
            imp[no.atributo] += (no.n / n_raiz) * no.ganho
            caminhar(no.esq)
            caminhar(no.dir)

        caminhar(self.raiz)
        total = imp.sum()
        return imp / total if total > 0 else imp

    def imprimir(self, nomes_atributos=None, no: No | None = None, recuo: str = "") -> str:
        no = self.raiz if no is None else no
        nome = (
            f"x[{no.atributo}]"
            if nomes_atributos is None or no.folha
            else str(nomes_atributos[no.atributo])
        )
        if no.folha:
            return f"{recuo}--> {self.classes_[no.classe]} (n={no.n}, imp={no.imp:.3f})\n"
        s = f"{recuo}[{nome} <= {no.limiar:.4g}]  ganho={no.ganho:.4f}  n={no.n}\n"
        s += self.imprimir(nomes_atributos, no.esq, recuo + "   ")
        s += self.imprimir(nomes_atributos, no.dir, recuo + "   ")
        return s


# ---------------------------------------------------------------------------
# 3. Regras SE-ENTAO
# ---------------------------------------------------------------------------


def extrair_regras(no: No, nomes_atributos, classes, prefixo=None):
    """Percorre a arvore e devolve regras SE-ENTAO com cobertura e confianca.

    cobertura  = n_folha / n_raiz     (fracao do treino que a regra explica)
    confianca  = p(classe | folha)    (pureza da folha)
    """
    if prefixo is None:
        prefixo = []
        extrair_regras._raiz_n = no.n  # guarda o total para a cobertura

    if no.folha:
        return [
            {
                "condicoes": list(prefixo),
                "classe": classes[no.classe],
                "n": no.n,
                "cobertura": no.n / extrair_regras._raiz_n,
                "confianca": float(no.prob[no.classe]),
            }
        ]

    nome = nomes_atributos[no.atributo] if nomes_atributos is not None else f"x[{no.atributo}]"
    esq = extrair_regras(no.esq, nomes_atributos, classes, prefixo + [f"{nome} <= {no.limiar:.4g}"])
    dir_ = extrair_regras(no.dir, nomes_atributos, classes, prefixo + [f"{nome} > {no.limiar:.4g}"])
    return esq + dir_


def formatar_regras(regras, min_cobertura: float = 0.0, top: int | None = None) -> str:
    regras = [r for r in regras if r["cobertura"] >= min_cobertura]
    regras = sorted(regras, key=lambda r: (-r["cobertura"], -r["confianca"]))
    if top:
        regras = regras[:top]
    linhas = []
    for i, r in enumerate(regras, 1):
        cond = " E ".join(r["condicoes"]) if r["condicoes"] else "(sempre)"
        linhas.append(
            f"R{i}: SE {cond}\n"
            f"     ENTAO {r['classe']}   [cobertura={r['cobertura']:.1%}, "
            f"confianca={r['confianca']:.1%}, n={r['n']}]"
        )
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# 4. Poda por erro reduzido
# ---------------------------------------------------------------------------


def _erros_subarvore(no: No, X: np.ndarray, y: np.ndarray) -> int:
    if X.shape[0] == 0:
        return 0
    if no.folha:
        return int(np.sum(y != no.classe))
    m = X[:, no.atributo] <= no.limiar
    return _erros_subarvore(no.esq, X[m], y[m]) + _erros_subarvore(no.dir, X[~m], y[~m])


def _podar_no(no: No, X: np.ndarray, y: np.ndarray) -> None:
    if no.folha:
        return
    m = X[:, no.atributo] <= no.limiar
    _podar_no(no.esq, X[m], y[m])          # bottom-up: filhos primeiro
    _podar_no(no.dir, X[~m], y[~m])
    if X.shape[0] == 0:                     # nenhuma amostra de validacao chega aqui
        no.virar_folha()
        return
    err_folha = int(np.sum(y != no.classe))
    if _erros_subarvore(no, X, y) >= err_folha:
        no.virar_folha()


def podar_erro_reduzido(arvore: ArvoreDecisao, X_val, y_val) -> ArvoreDecisao:
    """Poda pos-inducao: substitui por folha toda subarvore que nao reduz o erro
    de validacao. Simplifica o modelo sem custo de acuracia."""
    X_val = np.asarray(X_val, dtype=float)
    _podar_no(arvore.raiz, X_val, arvore._codificar_y(y_val))
    return arvore