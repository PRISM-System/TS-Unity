from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


class Exp_Classification(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)