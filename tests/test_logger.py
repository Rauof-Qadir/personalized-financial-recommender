import logging

from src.utils import logger as logger_module


def test_logger_writes_to_file_and_reuses_handlers(tmp_path, monkeypatch):
    log_file = tmp_path / "application.log"
    monkeypatch.setattr(logger_module, "LOG_FILE", log_file)

    logger = logger_module.get_logger("test_logger_file_output")

    try:
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2

        same_logger = logger_module.get_logger("test_logger_file_output")
        assert same_logger is logger
        assert len(logger.handlers) == 2

        logger.info("Automated logging test successful")

        for handler in logger.handlers:
            handler.flush()

        assert log_file.exists()
        assert "Automated logging test successful" in log_file.read_text(
            encoding="utf-8"
        )
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
