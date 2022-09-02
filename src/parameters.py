import yaml

with open("../parameters.yaml") as fp: 
    PARAMETERS = yaml.load(fp, Loader=yaml.FullLoader)
