from mlperf_logging import mllog

mllogger = mllog.get_mllogger()


def _paddle_resnet_print(logger, key, val=None, metadata=None, stack_offset=3, namespace="paddle_mlperf"):
    logger(key=key, value=val, metadata=metadata, stack_offset=stack_offset, namespace=namespace)


def paddle_resnet_print_start(key, val=None, metadata=None):
    _paddle_resnet_print(mllogger.start, key, val, metadata)


def paddle_resnet_print_end(key, val=None, metadata=None):
    _paddle_resnet_print(mllogger.end, key, val, metadata)


def paddle_resnet_print_event(key, val=None, metadata=None):
    _paddle_resnet_print(mllogger.event, key, val, metadata)

