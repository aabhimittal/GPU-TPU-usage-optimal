"""Shared test helpers (not collected as tests)."""


class FakeBackend:
    """Injectable accelerator back-end returning a fixed reading."""

    def __init__(self, name, reading, available=True):
        self.name = name
        self._reading = reading
        self._available = available

    @property
    def available(self):
        return self._available

    def read(self, index=0):
        return self._reading
