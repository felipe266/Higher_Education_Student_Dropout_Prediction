"""
=============================================================================
 PCA APLICADO AO MNIST — demonstração didática
=============================================================================

Acompanha a aula sobre maldição da dimensionalidade e extração de
características (Bishop, Seç. 1.4 e Cap. 12).

O script produz cinco figuras:

  fig1_variancia.png    Espectro de autovalores e variância acumulada
  fig2_projecao_2d.png  Os dados projetados nas 2 primeiras componentes
  fig3_componentes.png  A imagem média e os primeiros autovetores
  fig4_reconstrucao.png Reconstrução dos dígitos com M crescente
  fig5_knn_vs_M.png     Acurácia e tempo do k-NN em função de M

Uso:
    python pca-mnist.py

Requisitos:
    numpy, matplotlib, scikit-learn

-----------------------------------------------------------------------------
LEMBRETE DE NOTAÇÃO (usada nos comentários do código inteiro):

  N  = número de amostras (linhas da matriz X) — quantas imagens temos
  D  = número de características (colunas de X) — quantos pixels por imagem
  M  = número de componentes principais mantidas, sempre com M <= D
  X  = matriz de dados, formato (N, D)
  Z  = dados projetados no espaço reduzido, formato (N, M)
  u_i = i-ésimo autovetor da matriz de covariância = i-ésima componente

A ideia central do PCA: girar o sistema de eixos de modo que o primeiro eixo
novo aponte na direção de maior espalhamento dos dados, o segundo na direção
de maior espalhamento restante (perpendicular ao primeiro), e assim por
diante. Depois jogamos fora os últimos eixos, onde quase não há variação.
=============================================================================
"""

import time                       # cronometragem do treino/teste do k-NN
import warnings                   # para silenciar avisos do fetch_openml
from pathlib import Path          # manipulação de caminhos independente de SO

import matplotlib
matplotlib.use("Agg")            # backend sem janela; remova para exibir
# ^ "Agg" desenha direto em arquivo PNG, sem abrir janela gráfica. Isso é o que
#   permite rodar o script em servidor, terminal remoto ou notebook sem display.

import matplotlib.pyplot as plt   # interface de plotagem
import numpy as np                # álgebra linear e arrays numéricos

from sklearn.decomposition import PCA               # o PCA propriamente dito
from sklearn.model_selection import train_test_split  # divisão treino/teste
from sklearn.neighbors import KNeighborsClassifier    # classificador k-NN

# -----------------------------------------------------------------------------
# Configuração
# -----------------------------------------------------------------------------
SEMENTE = 42              # semente do gerador aleatório: garante que rodar o
                          # script duas vezes produza exatamente as mesmas
                          # figuras e números (reprodutibilidade)

N_AMOSTRAS = 10_000       # subamostra do MNIST (10k de 70k)


SAIDA = Path("figuras")   # pasta onde as figuras serão gravadas
SAIDA.mkdir(exist_ok=True)  # cria a pasta; exist_ok evita erro se já existir

np.random.seed(SEMENTE)   # fixa a aleatoriedade global do NumPy

# Ajustes estéticos globais: dpi mais alto = figura mais nítida no projetor;
# fonte 9 evita que os rótulos fiquem gigantes nas figuras pequenas.
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})


# -----------------------------------------------------------------------------
# 1. Carregamento dos dados
# -----------------------------------------------------------------------------
def carregar_dados():
    """Carrega o MNIST. Se não houver rede, cai para o load_digits (8x8)."""
    try:
        # fetch_openml baixa o MNIST da internet (~55 MB) e guarda em cache
        from sklearn.datasets import fetch_openml

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # esconde avisos de versão do OpenML
            # as_frame=False -> devolve arrays NumPy em vez de DataFrame pandas
            dados = fetch_openml("mnist_784", version=1, as_frame=False)

        # dados.data tem formato (70000, 784): cada linha é uma imagem 28x28
        # "achatada" (flatten) numa única linha de 784 números.
        X = dados.data.astype(np.float64)   # converte para float (PCA exige)
        y = dados.target.astype(int)        # rótulos vêm como texto: "0".."9"
        nome = "MNIST (28x28)"

        # Subamostragem estratificada para a demonstração ficar rápida
        if N_AMOSTRAS < len(X):
            # stratify=y mantém a mesma proporção de cada dígito na amostra;
            # sem isso, poderíamos sortear muitos "1" e poucos "8" por azar.
            # Só o primeiro par de retornos interessa; o resto vai para "_".
            X, _, y, _ = train_test_split(
                X, y, train_size=N_AMOSTRAS, stratify=y, random_state=SEMENTE
            )

    except Exception as erro:
        # Plano B, galera: O scikit-learn traz embutido um conjunto menor de
        # dígitos 8x8 (D=64). Toda a lógica do script continua funcionando
        # só as figuras ficam com menos detalhe.
        print(f"[aviso] Não foi possível baixar o MNIST ({type(erro).__name__}).")
        print("[aviso] Usando o conjunto 'digits' (8x8) embutido no scikit-learn.\n")
        from sklearn.datasets import load_digits
        dados = load_digits()
        X = dados.data.astype(np.float64)
        y = dados.target.astype(int)
        nome = "digits (8x8)"

    # Normaliza a intensidade para [0, 1].
    #
    # NOTA IMPORTANTE sobre padronização: aqui NÃO usamos z-score. Todos os
    # pixels compartilham a mesma unidade (intensidade luminosa), então não há
    # o problema de escala que justificaria padronizar. Pior: dividir pelo
    # desvio-padrão amplificaria os pixels de borda, que são quase sempre pretos
    # e carregam só ruído. Use z-score quando as colunas tiverem unidades
    # DIFERENTES (ex.: salário em reais e idade em anos).
    X = X / X.max()   # X.max() vale 255 no MNIST e 16 no digits

    # Testem utilizando a NORMALIZAÇÃO Min-Max...

    return X, y, nome


# -----------------------------------------------------------------------------
# 2. Espectro de autovalores e variância acumulada
# -----------------------------------------------------------------------------
def figura_variancia(pca, D):
    # explained_variance_ratio_[i] = fração da variância total capturada pela
    # componente i. É o autovalor lambda_i dividido pela soma de todos eles.
    # O vetor já vem ordenado do maior para o menor.
    var = pca.explained_variance_ratio_

    # Soma acumulada: acum[i] = variância explicada pelas i+1 PRIMEIRAS
    # componentes juntas. Sempre crescente, terminando em 1.0 (=100%).
    acum = np.cumsum(var)

    # Um "canvas" com dois painéis lado a lado (1 linha, 2 colunas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

    # --- Painel esquerdo: quanta variância cada componente explica ---
    # np.arange(1, len+1) numera as componentes de 1 até D (e não de 0 a D-1),
    # que é a convenção matemática usada em aula.
    ax1.plot(np.arange(1, len(var) + 1), var, lw=1.4, color="#1C4A78")

    # Escala log no eixo y: a primeira componente explica ~10% e a última
    # ~0.000001%. Em escala linear, tudo depois da componente 20 vira uma
    # linha colada no zero e não se vê o decaimento.
    ax1.set_yscale("log")
    ax1.set_xlabel("Componente $i$")          # $...$ ativa notação LaTeX
    ax1.set_ylabel("Fração da variância (escala log)")
    ax1.set_title("Espectro de autovalores")
    ax1.grid(alpha=0.3)                        # grade discreta ao fundo

    # --- Painel direito: variância acumulada + limiares usuais ---
    ax2.plot(np.arange(1, len(acum) + 1), acum, lw=1.8, color="#1C4A78")

    # Marca os três critérios mais usados na prática para escolher M.
    # (As três cores são iguais de propósito — o zip está aqui só para
    #  facilitar trocar por cores distintas se quiser.)
    for limiar, cor in zip([0.85, 0.95, 0.99], ["#C85020", "#C85020", "#C85020"]):
        # searchsorted acha a posição onde 'limiar' entraria no vetor ordenado
        # 'acum'; +1 converte índice base-0 em contagem de componentes.
        # Ou seja: o menor M tal que as M primeiras somam >= limiar.
        M = int(np.searchsorted(acum, limiar) + 1)

        ax2.axhline(limiar, ls="--", lw=0.8, color=cor, alpha=0.6)  # linha horiz.
        ax2.axvline(M, ls=":", lw=0.8, color=cor, alpha=0.6)        # linha vert.

        # Rótulo do tipo "95% -> M=154", deslocado um pouco do ponto de cruzamento
        # para não ficar em cima da curva (o 0.04*D escala com o eixo x).
        ax2.annotate(f"{limiar:.0%} → M={M}", xy=(M, limiar),
                     xytext=(M + 0.04 * D, limiar - 0.09), fontsize=8, color=cor)

    ax2.set_xlabel("Número de componentes $M$")
    ax2.set_ylabel("Variância acumulada")
    ax2.set_title(f"Quantas componentes bastam? (D = {D})")
    ax2.set_ylim(0, 1.05)     # folga acima do 100% para o rótulo não encostar
    ax2.grid(alpha=0.3)

    fig.tight_layout()        # ajusta margens para nada ficar cortado
    fig.savefig(SAIDA / "fig1_variancia.png", bbox_inches="tight")
    plt.close(fig)            # libera a memória da figura (importante em loops)

    return acum               # devolvido para reuso na função main()


# -----------------------------------------------------------------------------
# 3. Projeção nas duas primeiras componentes
# -----------------------------------------------------------------------------
def figura_projecao_2d(Z, y, pca):
    fig, ax = plt.subplots(figsize=(6, 5))

    # Cada ponto é uma imagem inteira, resumida em apenas 2 números:
    # Z[:, 0] = coordenada na PC1, Z[:, 1] = coordenada na PC2.
    # c=y colore o ponto pelo dígito verdadeiro repare que o PCA NÃO viu
    # os rótulos; se as cores se agrupam, é porque a estrutura já estava lá.
    disp = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=6,
                      alpha=0.65, linewidths=0)
    # s=6 -> pontos pequenos; alpha=0.65 -> semitransparentes, para enxergar
    # a densidade onde milhares de pontos se sobrepõem.

    barra = fig.colorbar(disp, ax=ax, ticks=range(10))  # legenda de cores 0..9
    barra.set_label("Dígito")

    # Informa no rótulo do eixo quanta variância cada eixo carrega no MNIST
    # as duas juntas dão ~17%, o que explica a mistura visível entre classes.
    v1, v2 = pca.explained_variance_ratio_[:2]
    ax.set_xlabel(f"PC 1 ({v1:.1%} da variância)")
    ax.set_ylabel(f"PC 2 ({v2:.1%} da variância)")
    ax.set_title("Espaço de alta dimensão comprimido em 2 eixos")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(SAIDA / "fig2_projecao_2d.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. Visualizando os autovetores ("autodígitos")
# -----------------------------------------------------------------------------
def figura_componentes(pca, lado):
    n = 15                     # quantos autovetores mostrar
    # Grade 2x8 = 16 quadros: 1 para a média + 15 para os autovetores
    fig, eixos = plt.subplots(2, 8, figsize=(10, 3.0))
    eixos = eixos.ravel()      # achata a matriz 2x8 em uma lista de 16 eixos,
                               # assim podemos indexar com eixos[0], eixos[1]...

    # Primeiro quadro: a imagem média (o ponto de origem do espaço centralizado)
    # pca.mean_ tem D valores; reshape devolve a forma de imagem (lado x lado).
    eixos[0].imshow(pca.mean_.reshape(lado, lado), cmap="gray")
    eixos[0].set_title("média", fontsize=8)
    eixos[0].axis("off")       # esconde eixos numéricos: é imagem, não gráfico

    # Demais quadros: os autovetores. Regiões claras e escuras indicam os
    # pixels que essa direção usa para diferenciar um dígito de outro.
    for i in range(n):
        comp = pca.components_[i].reshape(lado, lado)   # linha i = vetor u_{i+1}

        # Autovetores têm valores positivos E negativos. Centralizamos a escala
        # de cor em zero (vmin=-lim, vmax=+lim) para que o branco represente
        # exatamente 0 e o vermelho/azul indiquem os dois sinais simetricamente.
        lim = np.abs(comp).max()
        eixos[i + 1].imshow(comp, cmap="RdBu_r", vmin=-lim, vmax=lim)
        eixos[i + 1].set_title(f"$u_{{{i+1}}}$", fontsize=8)  # rende "u_1", "u_2"...
        eixos[i + 1].axis("off")

    fig.suptitle("Imagem média e primeiros autovetores da matriz de covariância",
                 fontsize=10)
    fig.tight_layout(h_pad=2.0)   # h_pad = espaço extra entre as duas linhas
    fig.savefig(SAIDA / "fig3_componentes.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Reconstrução com M crescente
# -----------------------------------------------------------------------------
def figura_reconstrucao(X, pca, lado, D, valores_M, n_exemplos=8):
    amostras = X[:n_exemplos]         # pega as primeiras 8 imagens do conjunto
    linhas = len(valores_M) + 1       # +1 porque a primeira linha é o original

    # figsize proporcional à grade: ~1 polegada por quadro, mantendo quadrados
    fig, eixos = plt.subplots(linhas, n_exemplos,
                              figsize=(1.05 * n_exemplos, 1.05 * linhas))

    # Linha de cima: as imagens originais
    for j in range(n_exemplos):
        eixos[0, j].imshow(amostras[j].reshape(lado, lado), cmap="gray")
        eixos[0, j].axis("off")
    eixos[0, 0].set_ylabel("original")   # (sem efeito visual: o eixo está off)

    # Demais linhas: projeta em M dimensões e volta ao espaço original
    for i, M in enumerate(valores_M, start=1):   # start=1 pula a linha 0
        # PASSO 1 — comprimir: transform centra os dados (subtrai a média) e
        # projeta nos D eixos; o fatiamento [:, :M] descarta tudo além de M.
        Z = pca.transform(amostras)[:, :M]

        # PASSO 2 — descomprimir: combinação linear dos M autovetores usando as
        # coordenadas Z como pesos, e soma-se a média de volta.
        # Formatos: (n, M) @ (M, D) -> (n, D). É a fórmula
        #     x_reconstruido ≈ média + soma_{i=1..M} z_i * u_i
        recon = Z @ pca.components_[:M] + pca.mean_

        # Erro quadrático médio pixel a pixel: mede o que se perdeu. Pela teoria,
        # ele é igual à soma dos autovalores DESCARTADOS (as componentes M+1..D).
        erro = np.mean((amostras - recon) ** 2)
        var_ret = pca.explained_variance_ratio_[:M].sum()   # variância mantida

        for j in range(n_exemplos):
            eixos[i, j].imshow(recon[j].reshape(lado, lado), cmap="gray")
            eixos[i, j].axis("off")

        # Rótulo da linha, escrito à esquerda do primeiro quadro.
        # transform=...transAxes usa coordenadas relativas ao quadro:
        # x=-0.55 significa "55% da largura à esquerda da borda"; y=0.5 é o meio.
        eixos[i, 0].text(-0.55, 0.5,
                         f"M={M}\n{var_ret:.0%} var\nEQM={erro:.4f}",
                         transform=eixos[i, 0].transAxes,
                         fontsize=7, va="center", ha="right")

    # Rótulo da linha original, em negrito para destacar a referência
    eixos[0, 0].text(-0.55, 0.5, f"original\nD={D}",
                     transform=eixos[0, 0].transAxes,
                     fontsize=7, va="center", ha="right", fontweight="bold")

    fig.suptitle("Reconstrução: informação perdida ao descartar componentes",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(SAIDA / "fig4_reconstrucao.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 6. Efeito prático: acurácia e tempo do k-NN em função de M
# -----------------------------------------------------------------------------
def figura_knn(X_tr, X_te, y_tr, y_te, pca, D, valores_M):
    """Mostra a maldição na prática: reduzir D pode MELHORAR o classificador."""
    # Projetamos UMA VEZ nas D componentes e depois só fatiamos com [:, :M].
    # Isso evita recalcular a projeção a cada valor de M as M primeiras
    # colunas da projeção completa são exatamente iguais à projeção em M.
    Z_tr_full = pca.transform(X_tr)
    Z_te_full = pca.transform(X_te)

    acuracias, tempos = [], []   # listas que alimentarão o gráfico

    # Cabeçalho da tabela impressa no terminal.
    # O ">6" alinha à direita numa largura de 6 caracteres.
    print(f"\n{'M':>6} {'Var. retida':>12} {'Acurácia':>10} {'Tempo (s)':>11}")
    print("-" * 42)

    def medir(Xa, Xb):
        """Ajusta e avalia o k-NN, devolvendo acurácia e tempo mediano."""
        marcas = []
        acc = None
        for _ in range(3):                     # repete para reduzir ruído
            # k=3 vizinhos: classifica pelo voto majoritário dos 3 pontos de
            # treino mais próximos (distância euclidiana, por padrão).
            knn = KNeighborsClassifier(n_neighbors=3)

            t0 = time.perf_counter()           # relógio de alta precisão
            knn.fit(Xa, y_tr)                  # no k-NN o "treino" só memoriza
            acc = knn.score(Xb, y_te)          # aqui está o custo real: buscar
                                               # os vizinhos de cada teste
            marcas.append(time.perf_counter() - t0)

        # Mediana em vez de média: descarta o efeito de uma repetição que tenha
        # sido atrapalhada por outro processo do sistema operacional.
        return acc, float(np.median(marcas))

    # Varre os valores de M, do menor ao maior
    for M in valores_M:
        acc, dt = medir(Z_tr_full[:, :M], Z_te_full[:, :M])
        acuracias.append(acc)
        tempos.append(dt)
        var_ret = pca.explained_variance_ratio_[:M].sum()
        print(f"{M:>6} {var_ret:>11.1%} {acc:>10.4f} {dt:>11.3f}")

    # Linha de referência: k-NN no espaço original, sem PCA
    # É o número que interessa comparar: em geral algum M pequeno EMPATA ou
    # SUPERA este valor, gastando uma fração do tempo.
    acc_base, t_base = medir(X_tr, X_te)
    print("-" * 42)
    print(f"{D:>6} {'100.0%':>11} {acc_base:>10.4f} {t_base:>11.3f}  <- sem PCA")

    fig, ax1 = plt.subplots(figsize=(6.5, 4))

    # Curva 1 (eixo y da esquerda, azul): acurácia
    ax1.plot(valores_M, acuracias, "o-", color="#1C4A78", label="acurácia (com PCA)")
    ax1.axhline(acc_base, ls="--", lw=1.2, color="#1C4A78", alpha=0.5)  # referência
    ax1.annotate(f"sem PCA (D={D}): {acc_base:.3f}",
                 xy=(valores_M[0], acc_base), xytext=(valores_M[0] * 1.1, acc_base + 0.012),
                 fontsize=8, color="#1C4A78")
    ax1.set_xlabel("Número de componentes $M$")
    ax1.set_ylabel("Acurácia", color="#1C4A78")   # cor casa com a curva
    ax1.tick_params(axis="y", labelcolor="#1C4A78")

    # Escala log em x: os valores de M vão de ~4 a ~600. Em escala linear, os
    # pontos pequenos (onde a curva é mais interessante) ficam espremidos.
    ax1.set_xscale("log")
    ax1.grid(alpha=0.3)

    # twinx() cria um segundo eixo y compartilhando o mesmo eixo x. Necessário
    # porque acurácia (0 a 1) e tempo (segundos) têm escalas incomparáveis.
    ax2 = ax1.twinx()
    ax2.plot(valores_M, tempos, "s--", color="#C85020", alpha=0.8, label="tempo")
    ax2.axhline(t_base, ls=":", lw=1.2, color="#C85020", alpha=0.5)
    ax2.set_ylabel("Tempo de treino + teste (s)", color="#C85020")
    ax2.tick_params(axis="y", labelcolor="#C85020")

    ax1.set_title("k-NN: menos dimensões, mais acurácia e menos tempo")
    fig.tight_layout()
    fig.savefig(SAIDA / "fig5_knn_vs_M.png", bbox_inches="tight")
    plt.close(fig)

    return acuracias, acc_base


# -----------------------------------------------------------------------------
# Programa principal
# -----------------------------------------------------------------------------
def main():
    X, y, nome = carregar_dados()
    N, D = X.shape                    # desempacota o formato (linhas, colunas)
    lado = int(round(np.sqrt(D)))     # 784 -> 28; 64 -> 8 (imagens quadradas)

    print("=" * 60)
    print(f"Conjunto: {nome}")
    print(f"N = {N} amostras (linhas)")
    print(f"D = {D} características (colunas) = {lado}x{lado} pixels")
    print("=" * 60)

    # 25% dos dados reservados para teste. stratify=y preserva a proporção de
    # cada dígito nos dois conjuntos.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEMENTE
    )

    # IMPORTANTE: o PCA é ajustado APENAS no treino. Ajustar no conjunto
    # completo vazaria informação do teste e inflaria a performance.
    # Sem n_components, o PCA calcula TODAS as D componentes; escolhemos M
    # depois, por fatiamento, o que permite comparar vários M sem refazer nada.
    pca = PCA(random_state=SEMENTE)
    pca.fit(X_tr)     # calcula a média, a matriz de covariância e seus autovetores

    acum = figura_variancia(pca, D)

    # Tabela de "quanto custa manter X% da informação"
    print("\nComponentes necessárias por limiar de variância:")
    for limiar in [0.80, 0.85, 0.90, 0.95, 0.99]:
        M = int(np.searchsorted(acum, limiar) + 1)
        print(f"  {limiar:.0%} da variância -> M = {M:4d} "
              f"(compressão de {D/M:.1f}x)")   # D/M = fator de redução

    Z = pca.transform(X_tr)          # projeção completa do treino, formato (N, D)
    figura_projecao_2d(Z[:, :2], y_tr, pca)   # só as 2 primeiras colunas
    figura_componentes(pca, lado)

    # Escala os valores de M ao tamanho do conjunto, para funcionar tanto
    # com o MNIST (D=784) quanto com o digits (D=64).
    # As frações crescem geometricamente (0.5%, 1%, 2%, ...) para amostrar bem
    # o gráfico em escala log. O set{} elimina duplicatas que surgem quando
    # duas frações arredondam para o mesmo inteiro (comum quando D é pequeno).
    valores_M = sorted({max(1, int(D * f)) for f in
                        [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]})
    valores_M = [M for M in valores_M if M <= D]   # segurança: nunca M > D

    # Para a figura de reconstrução usamos no máximo 5 valores, e ignoramos
    # M=1 (uma única componente produz imagens praticamente idênticas).
    figura_reconstrucao(X_te, pca, lado, D,
                        valores_M=[m for m in valores_M if m >= 2][:5])

    acuracias, acc_base = figura_knn(X_tr, X_te, y_tr, y_te, pca, D, valores_M)

    # argmax devolve a POSIÇÃO do maior valor na lista de acurácias; usamos essa
    # posição para recuperar o M correspondente.
    melhor = valores_M[int(np.argmax(acuracias))]
    print(f"\nMelhor acurácia com M = {melhor} "
          f"({max(acuracias):.4f} vs {acc_base:.4f} sem PCA)")
    print(f"\nFiguras salvas em: {SAIDA.resolve()}")   # caminho absoluto


# Este bloco só executa quando o arquivo é rodado diretamente
# (python pca-mnist.py). Se alguém importar este arquivo como módulo, as
# funções ficam disponíveis mas nada é executado automaticamente.
if __name__ == "__main__":
    main()