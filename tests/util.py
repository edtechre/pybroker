from typing import NamedTuple


class LazyFixture(NamedTuple):
    name: str


def get_fixture(request, param):
    if isinstance(param, LazyFixture):
        return request.getfixturevalue(param.name)
    return param
