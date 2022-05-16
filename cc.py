from model import *

k = BasicModel(256,256,1,97)

t = torch.randn(1,1,256,256)

b = k(t)
print(b.size())


# a = torch.rand(97,2)
# print(len(a.size()))

# a = torch.unsqueeze(a,0)
# print(a.size())
# print(a[0].size())
