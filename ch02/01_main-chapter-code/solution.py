class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        if x != self.p.get(x, x):
            self.p[x] = self.find(self.p.get(x, x))
        return self.p.get(x, x)

    def union(self, a, b):
        pa, pb = map(self.find, [a, b])
        if pa != pb:
            self.p[pa] = pb


from collections import defaultdict


class Solution:
    def numBusesToDestination(
        self, routes: list[list[int]], source: int, target: int
    ) -> int:
        if source == target:
            return 0
        uf = UF()
        inf = float("inf")
        inds = defaultdict(set)
        for ind, route in enumerate(routes):
            for a, b in zip(route, route[1:]):
                inds[a].add(ind)
                inds[b].add(ind)
                uf.union(a, b)

        g = defaultdict(list)

        n = len(routes)
        color = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                pa, pb = map(uf.find, [routes[i][0], routes[j][0]])
                if pa == pb:
                    g[i].append(j)
                    g[j].append(i)

        self.ans = inf
        self.cur = 1
        print(g, inds[target])

        def dfs(node):
            print(node, node in inds[target])
            if node in inds[target]:
                if self.cur < self.ans:
                    self.ans = self.cur
            color[node] = 1
            for nei in g[node]:
                if color[nei] == 0:
                    self.cur += 1
                    dfs(nei)
                    self.cur -= 1
            color[node] = 0

        for ind in inds[source]:
            dfs(ind)
        return self.ans if self.ans != inf else -1


s = Solution()
ans = s.numBusesToDestination([[1, 3], [2, 4, 5, 7], [3, 5]], 1, 4)
print(ans)
