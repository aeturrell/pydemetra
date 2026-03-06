from __future__ import annotations

from pydemetra.dictionary import user_defined


class TestUserDefined:
    def test_none_items(self):
        assert user_defined(object()) == {}

    def test_empty_items(self):
        assert user_defined(object(), items=[]) == {}
