---
layout: post
title: 正确使用随机数
categories: development
---


实际上，计算机几乎不可能实现真正的随机性，因为这些随机数字是由于确定的算法产生的，计算机生成的是伪随机数。
只不过这些看起来像随机的数，满足随机数的有许多已知的统计特性，在程序中被当作随机数使用。

### 随机数生成的基本原理

无论是 Go 标准库里面的 `math/rand`，还是 C++ 标准库中的 `std::default_random_engine`，其基本原理都相同，
都是使用 [`Lehmer` 在 1951 年提出的线性同余（linear congruential）的算法](https://en.wikipedia.org/wiki/Lehmer_random_number_generator)。
随机数序列 `x(1), x(2), ...` 由如下线性同余递推式子确定：

```
x(i+1) = A * x(i) % M
```

我们需要给定递推的初始值 `x(0)`, 这个值又叫做随机种子(seed)。
很显然，`x(0)`不能为 0，否找递推式会一直会生成 0。
观察上述式子，由于最后需要进行取模运算，生成过程中肯定会出现重复数字，存在某个固定周期。
有趣的是，当我们选取 `M` 为素数时，可以验证，总是存在某个 `A` 的取值使得周期为 `M-1`，即可以生成 `[1, M)` 中的每一个整数。
我们可以让 `M` 为一个比较大的素数  `2^31-1 = 2,147,483,647`，使得生成的序列的周期最大，而此时对应的 `A` 可以为 16807 或 48271。
例如，在 [go 的 `math/rand` 中选取的是数字 48271](https://cs.opensource.google/go/go/+/refs/tags/go1.20:src/math/rand/rng.go;l=186)。

### 避免频繁初始化随机数种子。 

随机数种子一般只需要在 `init()` 或 `main()` 中初始化一次。

```go
// Good:
func init() {
    rand.Seed(time.Now().UnixNano())
}

func isEven() bool {
    return rand.Intn(2) == 0
}
```

不要使用当前系统时间频繁初始化随机数种子，因为根据随机数生成的基本原理，这样做没有必要。

```go
// Bad:
func isEven() bool {
    rand.Seed(time.Now().UnixNano())
    return rand.Intn(2) == 0
}
```

- time.Now() 并不廉价。频繁获取时间是无谓的性能损耗。
- 如果在同一纳秒内调用这个方法，会产生完全一样的随机数。

从 go1.20 开始，[弃用了 `rand.Seed` 函数](https://pkg.go.dev/math/rand#Seed)，会自动初始化全局随机源，具体的讨论见 [golang 官方 issue](https://github.com/golang/go/issues/54880)。因此可以编写更简洁的代码：

```go
// Good: go1.20 and later versions
import math/rand
func isEven() bool {
    return rand.Intn(2) == 0
}
```
### 避免非均匀分布

直接使用 `rand.Intn(n)`, `rand.Int31n(n)` 或 `rand.Int61n()`, 获取 `[0, n)` 的随机数。

```go
// Good:
randNum := rand.Intn(n)
```

不要使用 `rand.Int()`,  `rand.Int31(n)` 或 `rand.Int61()` 对 n 取模来获取 `[0, n)` 的随机数。
因为如果 `n` 不能被 `m` 除尽，会导致分布偏移，无法形成均匀分布。

```go
// Bad:
randNum := rand.Int() % n
```

### 避免出现差一错误

例如抛掷一枚不均匀的硬币，有 80% 的概率得到正面， 有 20% 的概率得到反面。

```go
// Good:
type side bool

const (
	head = side(true)
	tail = side(false)
)

func tossCoin() side {
	return rand.Intn(100) < 80
}
```

`rand.Intn(100)` 返回的是 `0，1，..., 99` 之间的随机数， 如果将 `tossCoin()` 里面的表达式修改为 `rand.Intn(100) <= 80`，则会以 81% 的概率返回正面，导致经典的[差一错误Off-By-One Error](https://en.wikipedia.org/wiki/Off-by-one_error)。

```go
// Bad:
type side bool

const (
	head = side(true)
	tail = side(false)
)

func tossCoin() side {
	return rand.Intn(100) <= 80
}
```

### 注意线程安全和性能影响

rand.Seed 和取随机数的方法是线程安全的，但是这可能会导致性能问题。当需要高频大量生成随机数时，可能会造成大量的锁竞争。
因此，可以考虑使用新的 `rand.Source` 来创建新的随机数生成器。

```go
r := rand.New(rand.NewSource(seed)) // Caution: not goroutine-safe!
```

但是要注意， `rand.New()` 生成的随机数生成器不是并发安全的。
如果要实现并发安全，可以考虑在生成随机数的时候加锁保护。

```go
type SafeRand struct {
	r  *rand.Rand
	mu sync.Mutex
}

func NewSafeRand(seed int64) *SafeRand {
	return &SafeRand{
		r: rand.New(rand.NewSource(seed)),
	}
}

func (c *SafeRand) Intn(n int) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.r.Intn(n)
}
```

### 避免在安全领域，使用常规的非安全随机数生成器

`math/rand` 包不是密码学安全的，容易遭受攻击， 攻击者可能会根据生成的随机数的特征预测随机数。
因此永远不要使用 `math/rand` 生成对安全有敏感的随机数。
例如在某电商平台用户可以将商品链接分享给朋友，帮助其砍价，每为朋友可以砍价 0.00元 到 0.99 元。

```go
// Bad:
import "math/rand"
func genRandomMoneyFromMassBargin() int {
    return rand.Intn(100)
}
```

建议使用 [`crypto/rand`](https://pkg.go.dev/crypto/rand) 包中安全的随机数发生器来生成随机数。

```go
// Good:
import(
    "crypto/rand"
    "math/big"
)

func genRandomMoneyFromMassBargin() int {
	n, _ := crand.Int(crand.Reader, big.NewInt(100))
	return int(n.Int64())
}
```

### 参考

- [rand() Considered Harmful](https://learn.microsoft.com/en-us/events/goingnative-2013/rand-considered-harmful)



