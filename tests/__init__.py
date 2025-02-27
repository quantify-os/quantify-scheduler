def is_zhinst_available():
    try:
        import zhinst  # pyright: ignore[reportMissingImports]

        return True
    except ModuleNotFoundError:
        return False
