import bm25_search, dense_search

RRF_K = 60

def search(query, k=5, pool=50):
    bm = bm25_search.search(query, k=pool)
    de = dense_search.search(query, k=pool)
    fused, info = {}, {}
    for results in (bm, de):
        for rank, r in enumerate(results, start=1):
            cid = r[0]
            fused[cid] = fused.get(cid, 0) + 1 / (RRF_K + rank)
            info[cid] = r
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(cid, round(fused[cid], 4), info[cid][2], info[cid][3]) for cid in top]

if __name__ == "__main__":
    for r in search("stocks fell sharply before the holiday weekend"):
        print(r)