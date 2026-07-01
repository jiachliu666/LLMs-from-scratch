from collections import Counter
import heapq
from typing import List


class Solution:
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        cnt = Counter()
        available = list(range(n))
        using = []
        meetings.sort()

        for s, e in meetings:
            while using and using[0][1] <= s:
                heapq.heappush(available, heapq.heappop(using)[1])
            start_time = s
            try:
                room = heapq.heappop(available)
            except:
                start_time, room = heapq.heappop(using)
            cnt[room] += 1
            heapq.heappush(using, [start_time + e - s, room])
        max_usage = max(cnt.values(), default=0)

        for room in range(n):
            if cnt[room] == max_usage:
                return room


n = 3
meetings = [[1, 20], [2, 10], [3, 5], [4, 9], [6, 8]]

s = Solution()
s.mostBooked(n, meetings)
