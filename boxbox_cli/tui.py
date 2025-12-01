from __future__ import annotations

from typing import Dict, Optional, List, Tuple


def run_tui(context: Dict[str, Optional[object]] | None = None) -> None:
    """Run the curses-based TUI.

    - Top menu bar with items navigable via ← →
    - Enter to execute the selected menu's action (stub functions for now)
    - q or ESC to exit

    On Windows, you may need to: pip install windows-curses
    """
    context = context or {}

    # Import curses lazily to allow the CLI to be importable even if curses is missing.
    try:
        import curses
    except Exception as exc:  # pragma: no cover - environment dependent
        # Provide per-OS guidance for installing/using curses
        import platform
        system = platform.system()
        if system == "Windows":
            hint = "Install with: pip install windows-curses"
        elif system == "Darwin":
            hint = (
                "Ensure your Python includes ncurses (e.g., from python.org or Homebrew). "
                "If needed: brew install python"
            )
        else:  # Linux/other
            hint = "Ensure ncurses is installed (e.g., apt/yum/pacman install libncurses/terminfo)"
        print(
            "Curses is not available. "
            f"{hint}.\n"
            f"Original error: {exc}"
        )
        return

    from . import menu_actions as acts

    menu_items: List[Tuple[str, callable]] = [
        ("Live", acts.live_timing),
        ("Drivers", acts.drivers),
        ("Constructors", acts.constructors),
        ("Sessions", acts.sessions),
        ("Calendar", acts.calendar),
        ("Settings", acts.settings),
        ("Help", acts.help_about),
    ]

    def draw(stdscr, selected_idx: int, status: str) -> None:
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        # Initialize colors if supported
        has_colors = False
        try:
            if hasattr(curses, "has_colors") and curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                # Top bar background
                curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
                # Unused currently, reserved
                curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLUE)
                # Status/content area (white on black)
                curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
                # Menu label text (black on cyan) per request for high contrast
                curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_CYAN)
                has_colors = True
        except curses.error:
            has_colors = False

        # Top bar
        top_bar = " " * w
        try:
            if has_colors:
                stdscr.attron(curses.color_pair(1))
            stdscr.addstr(0, 0, top_bar)
            if has_colors:
                stdscr.attroff(curses.color_pair(1))
        except curses.error:
            stdscr.addstr(0, 0, top_bar)

        # Draw menu items centered (high-contrast colors; avoid bold to keep text crisp)
        label_parts = []
        for i, (label, _) in enumerate(menu_items):
            if i == selected_idx:
                label_parts.append(f" [ {label} ] ")
            else:
                label_parts.append(f"   {label}   ")
        labels = " ".join(label_parts)
        x = max(0, (w - len(labels)) // 2)
        try:
            if has_colors:
                # Use white-on-cyan for labels for better readability
                stdscr.attron(curses.color_pair(4))
            stdscr.addstr(0, x, labels[: max(0, w - x)])
            if has_colors:
                stdscr.attroff(curses.color_pair(4))
        except curses.error:
            try:
                stdscr.addstr(0, x, labels[: max(0, w - x)])
            except curses.error:
                stdscr.addstr(0, x, labels[: max(0, w - x)])

        # Status area (supports multi-line output). Use bold to simulate larger font.
        lines = (status or "").split("\n")
        max_rows = max(0, h - 3)
        for i, line in enumerate(lines[:max_rows]):
            status_line = line[: max(0, w - 2)]
            try:
                if has_colors:
                    stdscr.attron(curses.color_pair(3))
                stdscr.attron(curses.A_BOLD)
                stdscr.addstr(2 + i, 1, status_line)
                stdscr.attroff(curses.A_BOLD)
                if has_colors:
                    stdscr.attroff(curses.color_pair(3))
            except curses.error:
                try:
                    stdscr.attron(curses.A_BOLD)
                    stdscr.addstr(2 + i, 1, status_line)
                    stdscr.attroff(curses.A_BOLD)
                except curses.error:
                    pass

        # Help footer
        help_text = "← → navigate  •  Enter select  •  q or ESC to quit"
        try:
            stdscr.attron(curses.A_DIM)
            stdscr.addstr(h - 1, 1, help_text[: max(0, w - 2)])
            stdscr.attroff(curses.A_DIM)
        except curses.error:
            stdscr.addstr(h - 1, 1, help_text[: max(0, w - 2)])

        stdscr.refresh()

    def main(stdscr):
        # Basic curses setup
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)

        # Reduce ESC key delay on some terminals/platforms (best-effort)
        try:
            if hasattr(curses, "set_escdelay"):
                curses.set_escdelay(25)
        except Exception:
            pass

        selected = 0
        status = "Welcome to BoxBox CLI (framework)."
        draw(stdscr, selected, status)

        while True:
            ch = stdscr.getch()

            if ch in (ord("q"), 27):  # q or ESC
                break
            elif ch in (curses.KEY_RIGHT, ord("l")):
                selected = (selected + 1) % len(menu_items)
            elif ch in (curses.KEY_LEFT, ord("h")):
                selected = (selected - 1) % len(menu_items)
            elif ch in (curses.KEY_ENTER, 10, 13):
                # Execute the selected action
                try:
                    _, action = menu_items[selected]
                    status = action(context) or ""
                except Exception as e:  # pragma: no cover
                    status = f"Error: {e}"
            elif ch == curses.KEY_RESIZE:
                # Redraw on resize
                pass

            draw(stdscr, selected, status)

    curses.wrapper(main)
