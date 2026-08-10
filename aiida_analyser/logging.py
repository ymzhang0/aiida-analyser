import sys
from loguru import logger


def configure_logging(level="INFO"):
    logger.remove()

    logger.configure(
        extra={
            "analyser": "-",
            "node_pk": "-",
        }
    )

    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "{extra[analyser]: <20} | "
            "node={extra[node_pk]} | "
            "<level>{message}</level>"
        ),
    )