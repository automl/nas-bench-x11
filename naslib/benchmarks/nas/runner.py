import logging

from naslib.defaults.trainer import Trainer
from naslib.optimizers import RandomSearch, RegularizedEvolution, LocalSearch, Bananas, \
HB, BOHB, DEHB, REA_LCE, LS_LCE, Bananas_LCE, REA_SVR, LS_SVR, Bananas_SVR
from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, \
NasBench211SearchSpace, DartsSearchSpace, NasBenchNLPSearchSpace
from naslib.utils import utils, setup_logger, get_dataset_api

config = utils.get_config_from_args(config_type='nas')

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_optimizers = {
    'rs': RandomSearch(config),
    're': RegularizedEvolution(config),
    'bananas': Bananas(config),
    'ls': LocalSearch(config),
    'hb': HB(config),
    'bohb': BOHB(config),
    'dehb': DEHB(config),
    'ls_svr': LS_SVR(config),
    'rea_svr': REA_SVR(config),
    'bananas_svr': Bananas_SVR(config),
    'rea_lce': REA_LCE(config),
    'bananas_lce': Bananas_LCE(config),
    'ls_lce': LS_LCE(config),
}

supported_search_spaces = {
    'nasbench101': NasBench101SearchSpace(),
    'nasbench201': NasBench201SearchSpace(),
    'nasbench211': NasBench211SearchSpace(),    
    'darts': DartsSearchSpace(),
    'nlp': NasBenchNLPSearchSpace(),
}

dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)

trainer.search(resume_from="")
trainer.evaluate(resume_from="", dataset_api=dataset_api)