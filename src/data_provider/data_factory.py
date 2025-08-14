

def data_provider(args: BaseConfig, flag: str):
    factory = DataFactory(args)
    return factory.get_dataset(flag)