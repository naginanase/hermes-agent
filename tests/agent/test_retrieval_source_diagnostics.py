"""Retrieval diagnostics use standard DEBUG opt-in without inspecting disabled input."""
import logging

import agent.retrieval_sources as retrieval


def test_debug_opt_in_emits_only_safe_counts(caplog):
    value = {"output_text": "private prompt \ue200cite\ue202private_ref\ue201",
             "id": "private_response", "output": [{"type": "web_search_call", "action": {
                 "sources": [{"url": "https://secret.example/?token=credential", "title": "private title"}]}}]}
    with caplog.at_level(logging.DEBUG, logger=retrieval.logger.name):
        retrieval.log_retrieval_source_counts("receive.terminal", value)
    records = [record for record in caplog.records if record.name == retrieval.logger.name]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    message = records[0].getMessage()
    assert "boundary=receive.terminal" in message
    assert "source_count=1" in message
    assert "raw_citation_marker_count=1" in message
    assert "metadata_status=metadata_present" in message
    assert not any(secret in message for secret in ("private", "secret.example", "credential"))


def test_disabled_debug_does_not_inspect_input(caplog):
    class UnreadableInput:
        @property
        def output(self):
            raise AssertionError("disabled diagnostics inspected provider data")

    with caplog.at_level(logging.INFO, logger=retrieval.logger.name):
        retrieval.log_retrieval_source_counts("receive.terminal", UnreadableInput())
    assert not [record for record in caplog.records if record.name == retrieval.logger.name]


def test_unknown_boundary_is_not_logged_at_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger=retrieval.logger.name):
        retrieval.log_retrieval_source_counts("private-boundary", {"content": "private"})
    assert not [record for record in caplog.records if record.name == retrieval.logger.name]
