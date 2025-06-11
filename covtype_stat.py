       
from scipy.stats import rankdata
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import chi2

def compute_CD(avranks, n):
    k = len(avranks)
    q = [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
        2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
        3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
        3.426041, 3.458425, 3.488685, 3.517073,
        3.543799]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd

def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

def lloc(l, n):
    """
    List location in list of list structure.
    Enable the use of negative locations:
    -1 is the last element, -2 second last...
    """
    if n < 0:
        return len(l[0]) + n
    else:
        return n
    
def mxrange(lr):
    """
    Multiple xranges. Can be used to traverse matrices.
    This function is very slow due to unknown number of
    parameters.

    >>> mxrange([3,5])
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    >>> mxrange([[3,5,1],[9,0,-3]])
    [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

    """
    if not len(lr):
        yield ()
    else:
        # it can work with single numbers
        index = lr[0]
        if isinstance(index, int):
            index = [index]
        for a in range(*index):
            for b in mxrange(lr[1:]):
                yield tuple([a] + list(b))


# ----

def graph_ranks(avranks, names, cd, width=6, textspace=1, reverse=False, title=None, color='r'):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
    """
    
    width = float(width)
    textspace = float(textspace)

    sums = avranks
    tempsort = sorted([(a, i) for i, a in enumerate(sums)], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    lowv = min(1, int(math.floor(min(ssums))))
    highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4
    k = len(sums)
    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    def get_lines(sums, hsd):
        # get all pairs
        lsums = len(sums)
        allpairs = [(i, j)
                    for i, j in mxrange([[lsums], [lsums]]) if j > i]
        # remove not significant
        notSig = [(i, j) for i, j in allpairs
                    if abs(sums[i] - sums[j]) <= hsd]
        # keep only longest

        def no_longer(ij_tuple, notSig):
            i, j = ij_tuple
            for i1, j1 in notSig:
                if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                    return False
            return True

        longest = [(i, j) for i, j in notSig if no_longer((i, j), notSig)]

        return longest

    lines = get_lines(ssums, cd)
    linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

    # add scale
    distanceh = 0.25
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig, ax = plt.subplots(1,1,figsize=(width, height))
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom")

    k = len(ssums)

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=0.7)
        text(textspace - 0.2, chei, nnames[i], ha="right", va="center")

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=0.7)
        text(textspace + scalewidth + 0.2, chei, nnames[i],
             ha="left", va="center")

    # upper scale
    if not reverse:
        begin, end = rankpos(lowv), rankpos(lowv + cd)
    else:
        begin, end = rankpos(highv), rankpos(highv - cd)

    line([(begin, distanceh), (end, distanceh)], linewidth=0.7, color=color)
    line([(begin, distanceh + bigtick / 2),
            (begin, distanceh - bigtick / 2)],
            linewidth=0.7, color=color)
    line([(end, distanceh + bigtick / 2),
            (end, distanceh - bigtick / 2)],
            linewidth=0.7, color=color)
    text((begin + end) / 2, distanceh - 0.05, "CD",
            ha="center", va="bottom", color=color)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2
        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                    (rankpos(ssums[r]) + side, start)],
                    linewidth=2.5, color=color)
            start += height

    draw_lines(lines)
    
    return fig
    
def friedman_test(X, alpha=0.05):
    N = X.shape[0]
    k = X.shape[1]
    ranks = k + 1 - rankdata(X, axis=1)
    stat = (12 / N*k*(k+1)) * np.sum(np.sum(ranks, axis=0)**2) - 3*N*(k+1)
    chi = chi2.ppf(1 - alpha, k-1)
    return np.mean(ranks, axis=0), stat >= chi
    
#############################################################################

deltas = [1, 10, 20, 60]
frameworks = ['CR', 'TR-S', 'TR-U', 'TR-P']
labels = []
for d in deltas:
    for f in frameworks:
        labels.append('%s-%02d' % (f, d))
print(labels)

results = np.load('results/res_covtype_ova.npy')
print(results.shape) # classes, deltas (4), frameworks (4), chunks, metrics        

# ACC
res_acc = np.mean(results[:,:,:,:,0], axis=3)
print(res_acc.shape) # 7, 4, 4
res_acc = res_acc.reshape(7, -1) # 7, (deltas x frameworks)

print(friedman_test(res_acc))

ranks = []
for row in res_acc:
    ranks.append(rankdata(row).tolist())
ranks = np.array(ranks)

av_ranks = np.mean(ranks, axis=0)
cd = compute_CD(av_ranks, res_acc.shape[0])

fig = graph_ranks(av_ranks, labels, cd=cd, width=6, textspace=1.1, title='Balanced Accuracy ($BAC$)', color=plt.cm.coolwarm(.9))
plt.tight_layout()

plt.savefig("foo.png", dpi=300)
plt.savefig("fig_frameworks/CD_acc.png")
plt.savefig("fig_frameworks/CD_acc.eps")


# Label request
res_req = results[...,1]==0
n_chunks = res_req.shape[-1]

res_req = np.sum(res_req, axis=-1)/n_chunks
print(res_req.shape) # 7, 4, 4
res_req = res_req.reshape(7, -1) # 7, (deltas x frameworks)

print(friedman_test(res_req))


ranks = []
for row in res_req:
    ranks.append(rankdata(row).tolist())
ranks = np.array(ranks)

av_ranks = np.mean(ranks, axis=0)
cd = compute_CD(av_ranks, res_acc.shape[0])

fig = graph_ranks(av_ranks, labels, cd=cd, width=6, textspace=1.1, reverse=True, title='Label Request ($LReq$)', color=plt.cm.coolwarm(.9))
plt.tight_layout()

plt.savefig("foo.png", dpi=300)
plt.savefig("fig_frameworks/CD_req.png")
plt.savefig("fig_frameworks/CD_req.eps")


# CLF trainign
res_trn = results[...,2]==0
n_chunks = res_trn.shape[-1]

res_trn = np.sum(res_trn, axis=-1)/n_chunks
print(res_trn.shape) # 7, 4, 4
res_trn = res_trn.reshape(7, -1) # 7, (deltas x frameworks)

print(friedman_test(res_trn))

ranks = []
for row in res_trn:
    ranks.append(rankdata(row).tolist())
ranks = np.array(ranks)

av_ranks = np.mean(ranks, axis=0)
cd = compute_CD(av_ranks, res_acc.shape[0])

fig = graph_ranks(av_ranks, labels, cd=cd, width=6, textspace=1.1, reverse=True, title='Classifier Training Request ($TReq$)', color=plt.cm.coolwarm(.9))
plt.tight_layout()

plt.savefig("foo.png", dpi=300)
plt.savefig("fig_frameworks/CD_trn.png")
plt.savefig("fig_frameworks/CD_trn.eps")

