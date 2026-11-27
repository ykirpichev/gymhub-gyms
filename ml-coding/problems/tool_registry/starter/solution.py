# GYM_STARTER


class ToolRegistry:
    """Registry of callable tools with argument validation and an approval gate for side effects."""

    def __init__(self):
        pass

    def register(self, name, fn, schema, side_effect=False):
        """Add a tool; raise ValueError for a duplicate name or a schema that does not match fn."""
        pass

    def call(self, name, args, approved=False):
        """Validate, check approval, run the tool, and return a structured result dict."""
        pass
