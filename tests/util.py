def get_fixture(request, param):
    if isinstance(param, str):
        return request.getfixturevalue(param)
    return param
