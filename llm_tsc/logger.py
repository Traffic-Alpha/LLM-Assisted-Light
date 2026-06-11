"""Logging helper with an optional loguru dependency."""

import logging


try:
    from loguru import logger
except ModuleNotFoundError:
    logging.basicConfig(level=logging.INFO)

    class _Logger:
        def __init__(self) -> None:
            self._logger = logging.getLogger("llm_tsc")

        def _format(self, message, *args):
            return str(message).format(*args) if args else str(message)

        def debug(self, message, *args, **kwargs):
            self._logger.debug(self._format(message, *args))

        def info(self, message, *args, **kwargs):
            self._logger.info(self._format(message, *args))

        def warning(self, message, *args, **kwargs):
            self._logger.warning(self._format(message, *args))

        def error(self, message, *args, **kwargs):
            self._logger.error(self._format(message, *args))

    logger = _Logger()
