"""Input handler with Shift+Tab key binding for Plan Mode toggle.

Uses prompt_toolkit for input capture, allowing detection of special
key combinations like Shift+Tab. Rich remains responsible for all
output rendering -- prompt_toolkit only handles input.
"""

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style


class InputHandler:
    """Manages user input with mode-toggle key binding.

    Shift+Tab toggles between Normal mode and Plan mode.
    The prompt prefix changes to reflect the current mode.
    """

    def __init__(self):
        self._plan_mode: bool = False
        self._session: PromptSession | None = None
        self._bindings = self._setup_keybindings()

    def _setup_keybindings(self) -> KeyBindings:
        """Configure Shift+Tab to toggle mode."""
        bindings = KeyBindings()

        @bindings.add("s-tab")
        def _toggle_mode(event):
            self._plan_mode = not self._plan_mode
            # Exit prompt immediately so CLI detects the toggle
            event.app.exit(result="")

        return bindings

    @property
    def plan_mode(self) -> bool:
        return self._plan_mode

    @plan_mode.setter
    def plan_mode(self, value: bool) -> None:
        self._plan_mode = value

    def get_input(self) -> str:
        """Prompt for user input. Returns the entered text.

        The prompt shows current mode:
          Normal: "You> "  (green)
          Plan:   "Plan> " (magenta)

        Raises EOFError/KeyboardInterrupt on Ctrl+D/Ctrl+C.
        """
        if self._plan_mode:
            message = [
                ("class:plan-prompt", "Plan"),
                ("class:prompt-colon", "> "),
            ]
        else:
            message = [
                ("class:normal-prompt", "You"),
                ("class:prompt-colon", "> "),
            ]

        if self._session is None:
            self._session = PromptSession(key_bindings=self._bindings)

        result = self._session.prompt(message)
        return result.strip()

    def get_input_with_prompt(self, label: str, style: str = "class:plan-prompt") -> str:
        """Prompt for user input with a custom label (e.g. "Modifications").

        Shows: "Modifications> " and lets the user type on the same line.
        Raises EOFError/KeyboardInterrupt on Ctrl+D/Ctrl+C.
        """
        message = [
            (style, label),
            ("class:prompt-colon", "> "),
        ]
        if self._session is None:
            self._session = PromptSession(key_bindings=self._bindings)
        result = self._session.prompt(message)
        return result.strip()

    def select_option(self, options: list[tuple[str, str]]) -> int:
        """Interactive option selector with arrow keys and Enter.

        Args:
            options: List of (label, description) tuples.

        Returns:
            Selected index (0-based).
        """
        selected = [0]  # Mutable for closure

        def get_text_fragments() -> FormattedText:
            frags: FormattedText = []
            for i, (label, desc) in enumerate(options):
                if i == selected[0]:
                    frags.append(("class:selected", f"  > {label}"))
                else:
                    frags.append(("class:unselected", f"    {label}"))
                if desc:
                    frags.append(("class:description", f"  {desc}"))
                frags.append(("", "\n"))
            return frags

        bindings = KeyBindings()

        @bindings.add("up")
        def _(event):
            selected[0] = (selected[0] - 1) % len(options)

        @bindings.add("down")
        def _(event):
            selected[0] = (selected[0] + 1) % len(options)

        @bindings.add("enter")
        def _(event):
            event.app.exit(result=selected[0])

        @bindings.add("c-c")
        def _(event):
            event.app.exit(result=len(options) - 1)

        control = FormattedTextControl(get_text_fragments)
        layout = Layout(Window(content=control, height=len(options)))
        style = Style.from_dict({
            "selected": "bold fg:ansiwhite bg:ansiblue",
            "unselected": "",
            "description": "fg:ansigray",
        })
        app = Application(
            layout=layout,
            key_bindings=bindings,
            style=style,
            full_screen=False,
        )
        return app.run()
