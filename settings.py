import logging
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'console': {
            'format': '[%(asctime)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'stream': {
            'level': 'DEBUG',
            'formatter': 'console',
            'class': 'logging.StreamHandler',
        }
    },
    'loggers': {
        'notebook': {
            'handlers': ['stream'],
            'level': logging.DEBUG,
            'propagate': False
        },
        'deep_models': {
            'handlers': ['stream'],
            'level': logging.INFO,
            'propagate': False
        }
    }
}