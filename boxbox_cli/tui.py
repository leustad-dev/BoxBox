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
                # Dropdown background (simulate darker cyan by using cyan + dim attribute)
                curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)
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

    def select_year(
        stdscr,
        base_year: Optional[int] = None,
        *,
        include_next_year: bool = True,
        count_previous: int = 10,
        title_text: str = "Select Year",
        show_next_label: str = "Next Year",
        min_year: Optional[int] = None,
    ) -> Optional[int]:
        """Show a dropdown menu of years and return the chosen year.

        - include_next_year: if True, put next calendar year first (labeled by show_next_label)
        - count_previous: how many years to include starting from current year going backwards
        - title_text: title displayed at the top of the dropdown window

        Controls: Up/Down or j/k to move, Enter to select, ESC/q to cancel.
        Returns the selected year or None if canceled.
        """
        h, w = stdscr.getmaxyx()
        try:
            curses.start_color()
            curses.use_default_colors()
        except Exception:
            pass

        # Determine list of years based on options
        import datetime as _dt
        curr_year = _dt.date.today().year
        next_year = curr_year + 1
        base = curr_year if base_year is None else int(base_year)
        # Always compute relative to actual current year for consistent UI behavior
        prev_years = [curr_year - i for i in range(count_previous)]
        if min_year is not None:
            prev_years = [y for y in prev_years if y >= int(min_year)]
        years = ([next_year] if include_next_year else []) + prev_years
        labels_display = ([show_next_label] if include_next_year else []) + [str(y) for y in prev_years]
        sel_idx = 0

        # Dimensions for the dropdown window
        list_width = max(18, max(len(s) for s in labels_display) + 6)  # padding
        list_height = len(years) + 2  # borders/padding
        start_y = max(1, (h - list_height) // 3)
        start_x = max(2, (w - list_width) // 2)

        # Create a subwindow for the dropdown
        win = stdscr.derwin(list_height, list_width, start_y, start_x)
        win.keypad(True)

        def _draw_dropdown():
            try:
                win.erase()
            except Exception:
                pass
            # Frame
            try:
                win.box()
            except Exception:
                pass

            title = title_text
            try:
                win.addnstr(0, max(1, (list_width - len(title)) // 2), title, list_width - 2)
            except Exception:
                pass

            for i, y in enumerate(years):
                disp = labels_display[i]
                label = f"  {disp}  "
                try:
                    if curses.has_colors():
                        if i == sel_idx:
                            # Selected line: black on cyan + bold for visibility
                            win.attron(curses.color_pair(5))
                            win.attron(curses.A_BOLD)
                            win.addnstr(1 + i, 1, label.ljust(list_width - 2), list_width - 2)
                            win.attroff(curses.A_BOLD)
                            win.attroff(curses.color_pair(5))
                        else:
                            # Unselected: use dim attribute to simulate darker background effect
                            win.attron(curses.A_DIM)
                            win.addnstr(1 + i, 1, label.ljust(list_width - 2), list_width - 2)
                            win.attroff(curses.A_DIM)
                    else:
                        # No color support
                        marker = ">" if i == sel_idx else " "
                        win.addnstr(1 + i, 1, f"{marker} {disp}".ljust(list_width - 2), list_width - 2)
                except curses.error:
                    pass

            try:
                win.noutrefresh()
                stdscr.noutrefresh()
                curses.doupdate()
            except Exception:
                pass

        _draw_dropdown()

        while True:
            ch = stdscr.getch()
            if ch in (27, ord('q')):  # ESC or q
                return None
            elif ch in (curses.KEY_UP, ord('k')):
                sel_idx = (sel_idx - 1) % len(years)
                _draw_dropdown()
            elif ch in (curses.KEY_DOWN, ord('j')):
                sel_idx = (sel_idx + 1) % len(years)
                _draw_dropdown()
            elif ch in (curses.KEY_ENTER, 10, 13):
                return years[sel_idx]
            elif ch == curses.KEY_RESIZE:
                # Recompute geometry on resize
                h, w = stdscr.getmaxyx()
                list_height_new = len(years) + 2
                list_width_new = list_width
                start_y_new = max(1, (h - list_height_new) // 3)
                start_x_new = max(2, (w - list_width_new) // 2)
                try:
                    win.mvwin(start_y_new, start_x_new)
                    win.resize(list_height_new, list_width_new)
                except Exception:
                    pass
                _draw_dropdown()

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
        status = "Welcome to BoxBox CLI.\nA Formula 1 data tracker for the command line."
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
                    label, action = menu_items[selected]
                    if label == "Calendar":
                        # Show year selector with last 10 years; always base on current year
                        # so the dropdown consistently lists [current .. current-9]
                        chosen = select_year(stdscr, None)
                        if chosen is not None:
                            context["season"] = chosen
                        # After selection (or cancel), render calendar with current context
                        status = action(context) or ""
                    elif label == "Drivers":
                        # Show prior years but not before 2018; no next year entry
                        chosen = select_year(
                            stdscr,
                            None,
                            include_next_year=False,
                            count_previous=20,
                            title_text="Select Season",
                            min_year=2018,
                        )
                        if chosen is not None:
                            context["season"] = chosen
                        status = action(context) or ""
                    else:
                        status = action(context) or ""
                except Exception as e:  # pragma: no cover
                    status = f"Error: {e}"
            elif ch == curses.KEY_RESIZE:
                # Redraw on resize
                pass

            draw(stdscr, selected, status)

    curses.wrapper(main)
