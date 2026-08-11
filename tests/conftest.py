"""Shared safeguards for the offline EconBench test suite."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network_access(monkeypatch):
    """Fail clearly if an offline test attempts a network connection."""

    def blocked(*args, **kwargs):
        raise AssertionError(
            "Offline tests must not access the network. Use scripts/smoke_models.py "
            "for live provider checks."
        )

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
