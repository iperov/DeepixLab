class g_log:
    @staticmethod
    def set_level(level : int):
        """
        set console log level

        ```
            0   prints nothing
            1   only critical errors/warning
            2   more details for debugging
        ```
        """
        g_log._LEVEL = level

    @staticmethod
    def get_level() -> int:
        return g_log._LEVEL

    _LEVEL = 1