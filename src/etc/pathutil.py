import os

p2d = './1348'
trg = './paths'

if not os.path.exists(trg):
	os.mkdir(trg)

paths = []

for subdir in os.listdir(p2d):
	subdir = os.path.join(p2d, subdir, 'paths')
	if os.path.isdir(subdir):
		for fp in os.listdir(subdir):
			if fp[-3:] == 'png':
				fp = os.path.join(subdir, fp)
				with open(fp, 'rb') as f:
					data = f.read()
					paths.append((os.path.basename(fp), data))


for fp, data in paths:
	with open(os.path.join(trg, fp), 'wb') as f:
		f.write(data)

