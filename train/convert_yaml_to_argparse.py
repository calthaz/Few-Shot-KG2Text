import train.utils
print("################# Code to add in train.py #################")
print("parser = argparse.ArgumentParser()")
config = train.utils.read_configuration("train/config.yaml")

for key, value in config.items():
    print("parser.add_argument('--{}', type={}, required=True)".format(key, type(value).__name__))

print('args = parser.parse_args()')
print("config=\{\}")
for key, value in config.items():
    print("config['{}']=args.{}".format(key, key))


print("################# Arguments for Vertex #################")
for key, value in config.items():
    if(isinstance(value, str) and (" " in value)):
        value = '\"'+value+'\"'
    print("--{} {}".format(key, value))

print("################# commend line input #################")
for key, value in config.items():
    if(isinstance(value, str) and (" " in value)):
        value = '\"'+value+'\"'
    print("--{} {}".format(key, value), end=" ")

