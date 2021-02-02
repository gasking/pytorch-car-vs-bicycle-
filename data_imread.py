import yaml
def imread():
    t=open("cus.yml",'r+')
    files=yaml.load(t)
    return files