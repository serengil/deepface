from deepface.commons.logger import Logger

logger = Logger()


def test_singleton_same_object():
    assert Logger() == Logger()
    logger.info("✅ id's of instances of \"singletoned\" class Logger are the same")


def test_warn_show_once(monkeypatch):
    logs = []
    monkeypatch.setattr(logger, "dump_log", logs.append)
    monkeypatch.setattr(logger, "_shown_warnings", set())

    # default behaviour logs every time
    logger.warn("repeated warning")
    logger.warn("repeated warning")
    assert len(logs) == 2

    # show_once logs each distinct message only once
    logger.warn("once warning", show_once=True)
    logger.warn("once warning", show_once=True)
    logger.warn("another once warning", show_once=True)
    assert len(logs) == 4
    assert "once warning" in logs[2]
    assert "another once warning" in logs[3]

    monkeypatch.undo()
    logger.info("✅ warn with show_once logs a message only once")
