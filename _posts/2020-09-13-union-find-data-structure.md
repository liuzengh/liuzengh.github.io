---
layout: post
title: 并查集:用树结构表示不相交集合
categories: algorithm
---

一些应用涉及将n个不同的元素分成一组不相交的集合,这些应用经常需要进行两种特别的操作:寻找包含给定元素的唯一集合和合并两个集合。
并查集用有根树的数据结构来表示不相交集合,在查找时使用了路径压缩和按秩合并的启发式策略,可以获得一个几乎与总的操作数呈线性关系的运行时间。

### 并查集上的操作

- Make-Set(x): 建立一个新的集合，它的唯一成员是 x。因为各个集合是不相交的，故 x 不会出现在别的集合里面
- Find-Set(x)：确定元素 x 属于哪一个子集。它可以被用来确定两个元素是否属于同一子集
- Union(x, y)：将元素 x 和元素 y 所属的两个子集合并成同一个集合

由于支持后两种操作，一个不相交集也常被称为联合-查找数据结构（Union-find Data Structure）或合并-查找集合（Merge-find Set）。
为了更加精确的定义这些方法，需要定义如何表示集合。
一种常用的策略是为每个集合选定一个固定的元素，称为代表元，以表示整个集合。

### 建立集合

```
// 数组 p 存放元素 i 的父节点
// 初始化时元素的本身为集合代表元，指向自己，这是一个自环
for(int i=0; i<n; ++i){
	p[i] = i;
}
```

### 查找算法

如果把 x 的父节点保存在 p[x] 中（如果 x 没有父结点，则 p[x] 等于 x\），则可以写出查找结点 x 所在树的根节点的递归程序：

```
int find(int x){
// 如果 p[x] 等于 x，说明 x 本身就是树根，因此返回 x
// 否则返回x的父节点 p[x] 所在树的树根
	return p[x]==x ? x : find(p[x]);
}
```

### 按秩合并

显而易见的做法是，使具有较少结点的树的根指向具有较多结点的树的根。
这里并不显式地记录每个结点为根地子树地大小，而是使用一种易于分析地方法。
对于每个结点，维护一个秩，它表示该结点高度地一个上界。
使用按秩合并策略的Union操作中，我们可以让具有较小秩的根指向具有较大秩的根。

```
1. Make-Set(x)
2. 	x.p = x
// 初始化单个结点的秩为0
3. 	x.rank = 0
4. Union(x, y)
// Find-Set 操作不改变任何秩
5. 	Link(Find-Set(x), Find-Set(y))  
6. Link(x, y)
7. 	if x.rank > y.rank  
// 让较大秩的根成为较小秩的根的父节点， 但秩本身保持不变
8. 		y.p = x 
9. 	else
10.		x. p = y
// 两个根有相同秩时，任意选择一个作为父节点，并使它的秩加1
11.		if x.rank == y.rank  
12.			y.rank = y.rank + 1 
```

### 路径压缩

在特殊情况下，这棵树可能时一条长长的链。
设链的最后一个结点为 x，则每次执行 find(x) 都会遍历整条链，效率十分低下。
由于每棵树表示的只是一个集合，因此树的形态是无关紧要的，并不需要在“查找”操作之后保持树的形态不变，只需要把遍历过的结点都改成树根的子节点，下次操作就会块很多：

```
int find(int x){
// p[x]=x, 遍历过的结点都改成树根的子节点
	return p[x]==x ? x : p[x]=find(p[x]);
}
```


![并查集上的路径压缩](https://media.githubusercontent.com/media/liuzengh/liuzengh.github.io/main/images/post/2020-09-13/path-compress-ufds.png)
并查集上的路径压缩

### 应用：确定无向图的连通分量

不相交集合数据结构的许多应用之一是确定无向图的连通分量, Connected-Components过程使用不相交集合操作来计算一个图的连通分量。一旦Connected-Components预处理了该图，过程Same-Components就回答了两个顶点是否在同一个连通分量的询问。
当图的边集是静态时，我们可以使用深度优先搜索（或宽度优先搜索）来快速地计算连通分量。
然而，有时候边是动态被加入地，我们需要在加入每条边时，对连通分量进行维护。
在这种情况下，这里给定地实现比对于每个新边都运行一次新地深度优先搜索要高效得多。

```
1. Connected-Components(G)
2. 	for each vetex v \in G.V
3. 		Make-Set(x)
4. 	for each edge(u, v) \in G.E
5. 		if Find-Set(u) \neq Find-Set(v)
6. 			Union(u, v)
```

```
1. Same-Components(u, v)
2. 	if Find-Set(u) \neq Find-Set(v)
3. 		return True
4.	else	return False 
```

> 结论1： Connected-Components 处理完所有的边后，两个顶点在相同的连通分量中当且仅当它们在同一个集合中。

> 结论2：Connected-Components 作用于一个有 k 个连通分量的无向图 `G=(V,E)` 的过程中，Find-Set 需要调用 2|E| 次，Union 需要调用 |V|-k 次。

### 练习题

- [128.最长连续序列](#128最长连续序列)
- [130.被围绕的区域](#130.被围绕的区域)
- [399.除法求值](#399.除法求值)
- [547.朋友圈](#547.朋友圈)
- [684.冗余连接](#684.冗余连接)
- [721.账户合并](#721.账户合并)
- [765.情侣牵手](#765.情侣牵手)
- [924.尽量减少恶意软件的传播](#924.尽量减少恶意软件的传播)
- [947.移除最多的同行或同列石头](#947.移除最多的同行或同列石头)
- [990.等式方程的可满足性](#990.等式方程的可满足性)
- [1202.交换字符串中的元素](#1202.交换字符串中的元素)
- [1319.连通网络的操作次数](#1319.连通网络的操作次数)


<p id="128. 最长连续序列">128. 最长连续序列 </p>

```
给定一个未排序的整数数组，找出最长连续序列的长度。
要求算法的时间复杂度为 O(n)。
示例: 输入: [100, 4, 200, 1, 3, 2]; 输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        def find(x, p):
            if p[x] != x:
                p[x] = find(p[x], p)
            return p[x]
        
        s = set(nums)
        d = {}
        p = {item:item for item in s}

        for num in nums:
            if (num+1 in s):
                x = find(num, p)
                y =  find(num+1, p)
                if x!=y:
                    p[x] = y
            
        for item in p.keys():
            x = find(item, p)
            if x in d:
                d[x] += 1
            else:
                d[x] = 1
             
        return max(d.values())
```
<p id="130.被围绕的区域"> 130.被围绕的区域 </p>

```
给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。
找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
示例:
X X X X
X O O X
X X O X
X O X X
运行你的函数后，矩阵变为：
X X X X
X X X X
X X X X
X O X X

解释: 被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，
或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相
连”的。
```
```python
 # DFS
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # 边界情况：board = []
        def dfs(G, x, y, ch):
            G[x][y] = ch
            n, m = len(G), len(G[0])    
            dxdy = [(-1, 0), (0, -1), (1, 0), (0, 1)]
            for (dx, dy) in dxdy:
                newx, newy = x+dx, y+dy
                if 0<=newx<n and 0<=newy<m and G[newx][newy]=='O':
                    dfs(G, newx, newy, ch)
        if len(board) == 0:
            return
        n, m = len(board), len(board[0])=
        for i in range(m):
            if board[0][i] == 'O':
                dfs(board, 0, i, '?')
            if board[n-1][i] == 'O':
                dfs(board, n-1, i, '?')
        for j in range(n):
            if board[j][0] == 'O':
                dfs(board, j, 0, '?')
            if board[j][m-1] == 'O':
                dfs(board, j, m-1, '?')
        for i in range(1, n-1):
            for j in range(1, m-1):
                if board[i][j] == 'O':
                    dfs(board, i, j, 'X')
        for i in range(n):
            for j in range(m):
                if board[i][j] == '?':
                    board[i][j] = 'O'
```
<p id="399.除法求值">399.除法求值</p>

```
给出方程式 A / B = k, 其中 A 和 B 均为用字符串表示的变量， k 是一个浮点型数字。
根据已知方程式求解问题，并返回计算结果。如果结果不存在，则返回 -1.0。
示例 :
给定 a / b = 2.0, b / c = 3.0
问题: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? 
返回 [6.0, 0.5, -1.0, 1.0, -1.0 ]
输入为: 
vector<pair<string, string>> equations, vector<double>& values, 
vector<pair<string, string>> queries(方程式，方程式结果，问题方程式)，
其中 equations.size() == values.size()，即方程式的长度与方程式结果长度相等（程式与结果一一对应)，
并且结果值均为正数。以上为方程式的描述。返回vector<double>类型。
基于上述例子，输入如下：
equations(方程式) = [ ["a", "b"], ["b", "c"] ],
values(方程式结果) = [2.0, 3.0],
queries(问题方程式) = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ]. 
输入总是有效的。你可以假设除法运算中不会出现除数为0的情况，且不存在任何矛盾的结果。
```
```python
class Solution:
    def calcEquation(self, equations: List[List[str]], 
    values: List[float], queries: List[List[str]]) -> List[float]:
        def find(x, p, w):
            if x not in p:
                return ("", -1.0)
            else:
                result = 1.0
                while x != p[x]:
                    result *= w[x]
                    x = p[x]
                return (x, result)
        
        p, w = {}, {}
        for e in equations:
            p[e[0]], p[e[1]] = e[0], e[1]
            w[e[0]], w[e[1]] = 1.0, 1.0
            
        for (e, v) in zip(equations, values):
            x = find(e[0], p, w)
            y = find(e[1], p, w)
            if x[0]!=y[0] and x[0]!="" and y[0]!="":
                p[x[0]] = y[0]
                # root_x / root_y = (x/y) * (y/root_b) / (x/root_a)
                w[x[0]] = v * y[1] / x[1]
                
        ret = []
        for q in queries:
            x, y = q
            px, py = find(x, p, w), find(y, p, w)
            if px[0] == "" or py[0]=="" or px[0] != py[0]:
                ret.append(-1.0)
            else:
                # x/y = (x/root_x) / (y/root_y)
                ret.append(px[1]/py[1])
        return ret
```


<p id="547.朋友圈"> 547.朋友圈 </p>

 ```
班上有 N 名学生。其中有些人是朋友，有些则不是。他们的友谊具有是传递性。如果已知 A 是 B 的朋友，
B 是 C 的朋友，那么我们可以认为 A 也是 C 的朋友。所谓的朋友圈，是指所有朋友的集合。
给定一个 N * N 的矩阵 M，表示班级中学生之间的朋友关系。如果M[i][j] = 1，表示已知第 i 个和 j 个学生
互为朋友关系，否则为不知道。你必须输出所有学生中的已知的朋友圈总数。
示例 1：
输入：
[[1,1,0],
 [1,1,0],
 [0,0,1]]
输出：2 
解释：已知学生 0 和学生 1 互为朋友，他们在一个朋友圈。
第2个学生自己在一个朋友圈。所以返回 2 。
示例 2：
输入：
[[1,1,0],
 [1,1,1],
 [0,1,1]]
输出：1
解释：已知学生 0 和学生 1 互为朋友，学生 1 和学生 2 互为朋友，所以学生 0 和学生 2 也是朋友，
所以他们三个在一个朋友圈，返回 1 。
提示：
1 <= N <= 200
M[i][i] == 1
M[i][j] == M[j][i]
 ```
 ```cpp
 class Solution {
public:
    int find(int x, vector<int>& p){
        return p[x]==x ? x : p[x]=find(p[x], p);
    }
    int findCircleNum(vector<vector<int>>& M) {
        int n = M.size();
        vector<int> p(n);
        set<int> s;
        for(int i=0; i<n; ++i){
            p[i] = i;
        }
        for(int i=0; i<n; ++i){
            for(int j=0; j<i; ++j){
                if(M[i][j]){
                    int x = find(i, p);
                    int y = find(j, p);
                    if(x != y ){
                        p[x] = y;
                    }
                    //p[i] = j;
                }
            }
        }
       
        for(int i=0; i<n; ++i){
            s.insert(find(i, p));
        }
        return s.size();
    }
};
 ```

<p id="684.冗余连接">684.冗余连接 </p>

```
在本问题中, 树指的是一个连通且无环的无向图。
输入一个图，该图由一个有着N个节点 (节点值不重复1, 2, ..., N) 的树及一条附加的边构成。
附加的边的两个顶点包含在1到N中间，这条附加的边不属于树中已存在的边。结果图是一个以边组成的二维数组。
每一个边的元素是一对[u, v] ，满足 u < v，表示连接顶点u 和v的无向图的边。返回一条可以删去的边，
使得结果图是一个有着N个节点的树。如果有多个答案，则返回二维数组中最后出现的边。
答案边 [u, v] 应满足相同的格式 u < v。
示例 1：
输入: [[1,2], [1,3], [2,3]]
输出: [2,3]
解释: 给定的无向图为:
  1
 / \
2 - 3
示例 2：
输入: [[1,2], [2,3], [3,4], [1,4], [1,5]]
输出: [1,4]
解释: 给定的无向图为:
5 - 1 - 2
    |   |
    4 - 3
注意:
输入的二维数组大小在 3 到 1000。
二维数组中的整数在1到N之间，其中N是输入数组的大小。
```
```cpp
class Solution {
public:
    int find(int x, vector<int>& p){
        return p[x]==x ? x : p[x]=find(p[x], p);
    }
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        vector<int> p(n+1);
        for(int i=0; i<n+1; ++i){
            p[i] = i;
        }
        for(int i=0; i<n; ++i){
            int x = find(edges[i][0], p);
            int y = find(edges[i][1], p);
            if(x == y){
                return edges[i];
            }   
            else{
                p[x] = y;
            }
        }
        return vector<int>();
    }
};
```


<p id="721.账户合并"> 721.账户合并 </p>

```
给定一个列表 accounts，每个元素 accounts[i] 是一个字符串列表，
其中第一个元素 accounts[i][0]是名称 (name)，其余元素是 emails 表示该帐户的邮箱地址。
现在，我们想合并这些帐户。如果两个帐户都有一些共同的邮件地址，则两个帐户必定属于同一个人。
请注意，即使两个帐户具有相同的名称，它们也可能属于不同的人，因为人们可能具有相同的名称。
一个人最初可以拥有任意数量的帐户，但其所有帐户都具有相同的名称。
合并帐户后，按以下格式返回帐户：每个帐户的第一个元素是名称，其余元素是按顺序排列的邮箱地址。
accounts 本身可以以任意顺序返回。
例子 1:
Input: 
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], 
["John", "johnnybravo@mail.com"], ["John", "johnsmith@mail.com", 
"john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
Output: [["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  
["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
Explanation: 
  第一个和第三个 John 是同一个人，因为他们有共同的电子邮件 "johnsmith@mail.com"。 
  第二个 John 和 Mary 是不同的人，因为他们的电子邮件地址没有被其他帐户使用。
  我们可以以任何顺序返回这些列表，例如答案[['Mary'，'mary@mail.com']，
  ['John'，'johnnybravo@mail.com']，
  ['John'，'john00@mail.com'，'john_newyork@mail.com'，'johnsmith@mail.com']]仍然会被接受。
注意：
accounts的长度将在[1，1000]的范围内。
accounts[i]的长度将在[1，10]的范围内。
accounts[i][j]的长度将在[1，30]的范围内。
```
```python
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        def find(x, p):
            if x != p[x]:
                p[x] = find(p[x], p)
            return p[x]

        mails = set()
        mail_person = {}
        p = {}
        for account in accounts:
            for i in range(1, len(account)):
                mails.add(account[i])
                mail_person[account[i]] = account[0]
        
        for mail in mails:
            p[mail] = mail

        for account in accounts:
            
            for i in range(2, len(account)):
                x = find(account[1], p)
                y = find(account[i], p)
                if x != y:
                    p[y] = x
       # print(p)
        d = {} 
        for mail in mails:
            x = find(mail, p)
            if x in d:
                d[x].append(mail)
            else:
                d[x] = [mail]
      #  print(d)
        ret = []
        for key in d:
            ret.append(sorted([mail_person[key]] + d[key]))
        return ret
```
<p id="765.情侣牵手">765.情侣牵手 </p>

```
N 对情侣坐在连续排列的 2N 个座位上，想要牵到对方的手。 
计算最少交换座位的次数，以便每对情侣可以并肩坐在一起。 
一次交换可选择任意两人，让他们站起来交换座位。
人和座位用 0 到 2N-1 的整数表示，情侣们按顺序编号，第一对是 (0, 1)，第二对是 (2, 3)，以此类推，
最后一对是 (2N-2, 2N-1)。
这些情侣的初始座位  row[i] 是由最初始坐在第 i 个座位上的人决定的。
示例 1:
输入: row = [0, 2, 1, 3]
输出: 1
解释: 我们只需要交换row[1]和row[2]的位置即可。
示例 2:
输入: row = [3, 2, 0, 1]
输出: 0
解释: 无需交换座位，所有的情侣都已经可以手牵手了。
说明:
len(row) 是偶数且数值在 [4, 60]范围内。
可以保证row 是序列 0...len(row)-1 的一个全排列。
```

```python
class Solution:
    def minSwapsCouples(self, row: List[int]) -> int:
        d = {}
        def find(x, p):
            if x != p[x]:
                p[x] = find(p[x], p)
            return p[x]

        n = len(row) 
        p = [i for i in range(n)]
        for i in range(0, n, 2):
            p[row[i]] = p[row[i+1]]

        for i in range(0, n, 2):
            x = find(row[i], p)
            y = find(row[i]-1 if row[i]%2 else row[i]+1, p)
            if x != y:
                p[x] = y

            x = find(row[i+1], p)
            y = find(row[i+1]-1 if row[i+1]%2 else row[i+1]+1, p)
            if x != y:
                p[x] = y
        for v in p:
            x = find(v, p)
            if x in d:
                d[x] += 1
            else:
                d[x] = 1
        return sum(d.values())//2 - len(d)
            

```
<p id="924.尽量减少恶意软件的传播">924.尽量减少恶意软件的传播 </p>

```
在节点网络中，只有当 graph[i][j] = 1 时，每个节点 i 能够直接连接到另一个节点 j。
一些节点 initial 最初被恶意软件感染。
只要两个节点直接连接，且其中至少一个节点受到恶意软件的感染，那么两个节点都将被恶意软件感染。
这种恶意软件的传播将继续，直到没有更多的节点可以被这种方式感染。
假设 M(initial) 是在恶意软件停止传播之后，整个网络中感染恶意软件的最终节点数。
我们可以从初始列表中删除一个节点。如果移除这一节点将最小化 M(initial)， 则返回该节点。
如果有多个节点满足条件，就返回索引最小的节点。
请注意，如果某个节点已从受感染节点的列表 initial 中删除，它以后可能仍然因恶意软件传播而受到感染。
示例 1：
输入：graph = [[1,1,0],[1,1,0],[0,0,1]], initial = [0,1]
输出：0
示例 2：
输入：graph = [[1,0,0],[0,1,0],[0,0,1]], initial = [0,2]
输出：0
示例 3：
输入：graph = [[1,1,1],[1,1,1],[1,1,1]], initial = [1,2]
输出：1
提示：
1 < graph.length = graph[0].length <= 300
0 <= graph[i][j] == graph[j][i] <= 1
graph[i][i] == 1
1 <= initial.length < graph.length
0 <= initial[i] < graph.length
```
首先，把图中所有的连通分量各自标上不同的颜色，这可以用并查集或深度优先搜索来实现。如果 initial 中的两个节点的颜色相同（即属于同一个连通分量），那移除这种节点是不会减少 M(initial) 的，因为恶意软件会感染同一个连通分量中的所有节点。因此，对于 initial 中颜色唯一的节点，从中选择一个移除来最大限度地减少被感染节点数。(如果有多个节点都可以达成最优解，就选择下标最小的节点。另外，如果没有颜色唯一的节点，就直接返回下标最小的节点。)
```python
class Solution:
    def minMalwareSpread(self, graph: List[List[int]], initial: List[int]) -> int:
        def dfs(G, v, vis, count):
            for u in G[v]:
                if u not in vis:
                    count[0] += 1
                    vis.add(u)
                    dfs(G, u, vis, count)

        n = len(graph)
        G = [[] for i in range(n)]
        vis = set()
        for i in range(n):
            for j in range(n):
                if i!=j and graph[i][j] == 1:
                    G[i].append(j)
        count = [0]
        initial.sort()
        ret, value = None, n+1
        for u in initial:
            count = [0]
            for v in initial:
                if v!=u and v not in vis:
                    vis.add(v)
                    count[0] += 1
                    dfs(G, v, vis, count)
            #print(count[0])
            if count[0] < value:
                value = count[0]
                ret = u
            vis.clear()
        return ret
        
```
<p id="947.移除最多的同行或同列石头"> 947.移除最多的同行或同列石头 </p>

```
我们将石头放置在二维平面中的一些整数坐标点上。每个坐标点上最多只能有一块石头。
每次 move 操作都会移除一块所在行或者列上有其他石头存在的石头。
请你设计一个算法，计算最多能执行多少次 move 操作？
示例 1：
输入：stones = [[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]]
输出：5
示例 2：
输入：stones = [[0,0],[0,2],[1,1],[2,0],[2,2]]
输出：3
示例 3：
输入：stones = [[0,0]]
输出：0
提示：
1 <= stones.length <= 1000
0 <= stones[i][j] < 10000
```

```python
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        def dfs(row, col, vis, stones,  v, count):
            r, c = stones[v]
            for u in row[r]:
                if u not in vis:
                    count[0] += 1
                    vis.add(u)
                    dfs(row, col, vis, stones, u, count)

            for u in col[c]:
                if u not in vis:
                    count[0] += 1
                    vis.add(u)
                    dfs(row, col, vis, stones, u, count)
       
        d = {}
        vis = set()
        row, col = {}, {}
        for i in range(len(stones)):
            if stones[i][0] in row:
                row[stones[i][0]].append(i)
            else:
                row[stones[i][0]] = [i]
            if stones[i][1] in col:
                col[stones[i][1]].append(i)
            else:
                col[stones[i][1]] = [i]
        
        ret = 0
        for v in range(len(stones)):
            if v not in vis:
                count = [1]
                vis.add(v)
                dfs(row, col, vis, stones, v, count)
                ret += (count[0]-1)
               # print(count)

        return ret
```

<p id="990.等式方程的可满足性">990.等式方程的可满足性</p>

```
给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 equations[i] 的长度为 4，
并采用两种不同的形式之一："a==b" 或 "a!=b"。在这里，a 和 b 是小写字母（不一定不同），
表示单字母变量名。只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 true，
否则返回 false。 
示例 1：
输入：["a==b","b!=a"]
输出：false
解释：如果我们指定，a = 1 且 b = 1，那么可以满足第一个方程，但无法满足第二个方程。
没有办法分配变量同时满足这两个方程。
示例 2：
输入：["b==a","a==b"]
输出：true
解释：我们可以指定 a = 1 且 b = 1 以满足满足这两个方程。
示例 3：
输入：["a==b","b==c","a==c"]
输出：true
示例 4：
输入：["a==b","b!=c","c==a"]
输出：false
示例 5：
输入：["c==c","b==d","x!=z"]
输出：true
提示：
1 <= equations.length <= 500
equations[i].length == 4
equations[i][0] 和 equations[i][3] 是小写字母
equations[i][1] 要么是 '='，要么是 '!'
equations[i][2] 是 '='
```

```cpp
class Solution {
/*
   "a==b":字母a和b在同一个集合里面
    "a!=b":字母a和b不在同一个集合里面
    1. a==b:把字母a和b分到同一个集合里面
    2. a!=b:判断字母a和b是否在不同的集合里面, 若在同一个集合里面则返回false
*/
public:
    int find(int x, vector<int>& p){
        return p[x]==x ? x : find(p[x], p);
    }
    bool equationsPossible(vector<string>& equations) {
        vector<int> p(26);
        for(int i=0; i<26; ++i) p[i] = i;
        for(auto e=equations.begin(); e!=equations.end(); ++e){
            if((*e)[1] == '='){
                int x = find((*e)[0]-'a', p);
                int y = find((*e)[3]-'a', p);
                if(x !=y) p[x] = y;
            }
        }
        for(auto e=equations.begin(); e!=equations.end(); ++e){
            if((*e)[1] == '!'){
                int x = find((*e)[0]-'a', p);
                int y = find((*e)[3]-'a', p);
                if(x ==y){
                    return false;
                }
            }
        }
        return true;
    }
};
```


<p id="1202.交换字符串中的元素"> 1202.交换字符串中的元素 </p>

```
给你一个字符串 s，以及该字符串中的一些「索引对」数组 pairs，其中 pairs[i] = [a, b] 表示字符串中的
两个索引（编号从 0 开始）。
你可以 任意多次交换 在 pairs 中任意一对索引处的字符。
返回在经过若干次交换后，s 可以变成的按字典序最小的字符串。
示例 1:
输入：s = "dcab", pairs = [[0,3],[1,2]]
输出："bacd"
解释： 
交换 s[0] 和 s[3], s = "bcad"
交换 s[1] 和 s[2], s = "bacd"
示例 2：
输入：s = "dcab", pairs = [[0,3],[1,2],[0,2]]
输出："abcd"
解释：
交换 s[0] 和 s[3], s = "bcad"
交换 s[0] 和 s[2], s = "acbd"
交换 s[1] 和 s[2], s = "abcd"
示例 3：
输入：s = "cba", pairs = [[0,1],[1,2]]
输出："abc"
解释：
交换 s[0] 和 s[1], s = "bca"
交换 s[1] 和 s[2], s = "bac"
交换 s[0] 和 s[1], s = "abc"
提示：
1 <= s.length <= 10^5
0 <= pairs.length <= 10^5
0 <= pairs[i][0], pairs[i][1] < s.length
s 中只含有小写英文字母
```

```python
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        d = {}
        def find(x, p):
            if x != p[x]:
                p[x] = find(p[x], p)
            return p[x]

        n = len(s)
        ret = []
        t = n*[0]
        p = [i for i in range(n)]
        for pair in pairs:
            x, y = find(pair[0], p), find(pair[1], p)
            if x != y:
                if x<y:
                    p[y] = x
                else:
                    p[x] = y
        for v in range(len(p)):
            x = find(v, p)
            if x in d:
                d[x].append((s[v], v))
            else:
                d[x] = [(s[v], v)]
        for item in d.values():
            l1 = sorted(item, key=lambda x: x[0])
            l2 = sorted(item, key=lambda x: x[1])
            ret += [(l1[i][0], l2[i][1]) for i in range(len(l1))]
        #print(ret)
        for item in ret:
            t[item[1]] = item[0]
        return ''.join(t)
```
<p id="1319.连通网络的操作次数"> 1319.连通网络的操作次数 </p>

```
用以太网线缆将 n 台计算机连接成一个网络，计算机的编号从 0 到 n-1。线缆用 connections 表示，
其中 connections[i] = [a, b] 连接了计算机 a 和 b。
网络中的任何一台计算机都可以通过网络直接或者间接访问同一个网络中其他任意一台计算机。
给你这个计算机网络的初始布线 connections，你可以拔开任意两台直连计算机之间的线缆，
并用它连接一对未直连的计算机。
请你计算并返回使所有计算机都连通所需的最少操作次数。如果不可能，则返回 -1 。 
输入：n = 4, connections = [[0,1],[0,2],[1,2]]
输出：1
解释：拔下计算机 1 和 2 之间的线缆，并将它插到计算机 1 和 3 上。
输入：n = 6, connections = [[0,1],[0,2],[0,3],[1,2],[1,3]]
输出：2
输入：n = 6, connections = [[0,1],[0,2],[0,3],[1,2]]
输出：-1
解释：线缆数量不足。
输入：n = 5, connections = [[0,1],[0,2],[3,4],[2,3]]
输出：0
提示：
1 <= n <= 10^5
1 <= connections.length <= min(n*(n-1)/2, 10^5)
connections[i].length == 2
0 <= connections[i][0], connections[i][1] < n
connections[i][0] != connections[i][1]
没有重复的连接。
两台计算机不会通过多条线缆连接。
```
```cpp
class Solution {
public:
/*
1. 树的边 = n-1,所以电缆数必须大于等于计算机个数减一
2. 连通分量个数为m, 则只需要用m-1个电缆把这m个计算机组连接在一起就可以了
*/
    int find(int x, vector<int>& p){
        return p[x]==x ? x : p[x]=find(p[x], p);
    }
    int makeConnected(int n, vector<vector<int>>& connections) {
        vector<int> p(n);
        set<int> s;
        for(int i=0; i<n; ++i) p[i] = i;
        if(connections.size()<n-1){
            return -1;
        }
        for(auto connection=connections.begin(); 
        		connection!=connections.end(); ++connection){
            int x = find((*connection)[0], p);
            int y = find((*connection)[1], p);
            if(x != y){
                p[x] = y;
            }
        }
        for(auto item=p.begin(); item!=p.end(); ++item){
            s.insert(find(*item, p));
        }
        return s.size()-1;
    }
};
```

### 参考

1. 算法竞赛入门经典/刘汝佳编著.——2版.——北京：清华大学出版社，2014
2. 算法导论（原书第3版）/ （美）科尔曼(Cormen, T.H.)等著；殷建平等译. ——北京:机械工业出版社，2013.1
3. 力扣：https://leetcode-cn.com/