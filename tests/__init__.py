def is_zhinst_available():
    try:
        import zhinst  # noqa: F401 # pyright: ignore[reportMissingImports]

        return True
    except ModuleNotFoundError:
        return False
