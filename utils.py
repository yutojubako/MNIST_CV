import json

def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        for k, v in config_args.items():
            setattr(args, k, v)
    del args.config
    return args