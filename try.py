import glob


print('Named explicitly:')
for name in glob.glob(r'Road-Signs-Project\Road-Signs-Project\dataset2\test\[0-2]'):
    print(name)
