def Naturals(n):
    yield n
    yield from Naturals(n+1)

s = Naturals(1)
print("Natural #s", next(s), next(s), next(s), next(s))
def sieve(s):
    n = next(s)
    yield n
    yield from sieve(i for i in s if i%n != 0)
p = sieve(Naturals(2))

print("Prime #s", next(p), next(p), next(p), \
    next(p), next(p), next(p), next(p), next(p))

def gensend():
    item = yield
    yield item

g = gensend()
next(g)

print ( g.send("hello"))