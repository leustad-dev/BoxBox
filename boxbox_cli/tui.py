from __future__ import annotations

from typing import Dict, Optional, List, Tuple, Any
import threading
import time
import platform
import datetime as _dt


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
        print("Curses is not available. " f"{hint}.\n" f"Original error: {exc}")
        return

    from . import menu_actions as acts

    # Global-ish color state for this screen
    has_colors: bool = False
    _inactive_pair_bg: Optional[int] = None  # runtime-chosen color index for dark grey bg

    def _init_colors_if_needed():
        nonlocal has_colors
        nonlocal _inactive_pair_bg
        try:
            if hasattr(curses, "has_colors") and curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                # Top bar background
                curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
                # Reserved
                curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLUE)
                # Status/content area (white on black)
                curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
                # Menu label text (black on cyan)
                curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_CYAN)
                # Dropdown background (simulate darker cyan using dim on same pair)
                curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)
                # Loading notice colors
                try:
                    curses.init_pair(7, curses.COLOR_RED, -1)
                except curses.error:
                    curses.init_pair(7, curses.COLOR_RED, curses.COLOR_BLACK)
                curses.init_pair(8, curses.COLOR_RED, curses.COLOR_BLACK)

                # Inactive header background: try to create a dark grey background if terminal allows
                try:
                    if hasattr(curses, "can_change_color") and curses.can_change_color():
                        # Choose a custom color index that is likely free (avoid 0..7). Use 8.
                        grey_idx = 8
                        # Define a dark grey (approx 25%)
                        curses.init_color(grey_idx, 250, 250, 250)  # values 0-1000
                        _inactive_pair_bg = grey_idx
                        curses.init_pair(9, curses.COLOR_WHITE, grey_idx)  # labels white on dark grey
                        curses.init_pair(10, curses.COLOR_BLACK, grey_idx)  # alt: black on dark grey
                    else:
                        _inactive_pair_bg = None
                except curses.error:
                    _inactive_pair_bg = None
                has_colors = True
        except curses.error:
            has_colors = False

    menu_items: List[Tuple[str, callable]] = [
        ("Live", acts.live_timing),
        ("Drivers", acts.drivers),
        ("Results", acts.results),
        ("Sessions", acts.sessions),
        ("Calendar", acts.calendar),
        ("Settings", acts.settings),
        ("Help", acts.help_about),
    ]

    def draw(stdscr, selected_idx: int, status: str, *, focused: bool = True) -> None:
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        # Initialize colors once
        _init_colors_if_needed()

        # Top bar
        top_bar = " " * w
        try:
            if has_colors:
                if focused:
                    stdscr.attron(curses.color_pair(1))  # black on cyan
                else:
                    # inactive state: prefer dark grey if available, else dimmed cyan
                    if _inactive_pair_bg is not None:
                        stdscr.attron(curses.color_pair(9))  # white on dark grey
                    else:
                        stdscr.attron(curses.color_pair(1)); stdscr.attron(curses.A_DIM)
            stdscr.addstr(0, 0, top_bar)
            if has_colors:
                if focused:
                    stdscr.attroff(curses.color_pair(1))
                else:
                    if _inactive_pair_bg is not None:
                        stdscr.attroff(curses.color_pair(9))
                    else:
                        stdscr.attroff(curses.A_DIM); stdscr.attroff(curses.color_pair(1))
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
                if focused:
                    # Use black-on-cyan for labels (request)
                    stdscr.attron(curses.color_pair(4))
                else:
                    # Inactive: white on dark grey if available; else dimmed cyan
                    if _inactive_pair_bg is not None:
                        stdscr.attron(curses.color_pair(9))
                    else:
                        stdscr.attron(curses.color_pair(4)); stdscr.attron(curses.A_DIM)
            stdscr.addstr(0, x, labels[: max(0, w - x)])
            if has_colors:
                if focused:
                    stdscr.attroff(curses.color_pair(4))
                else:
                    if _inactive_pair_bg is not None:
                        stdscr.attroff(curses.color_pair(9))
                    else:
                        stdscr.attroff(curses.A_DIM); stdscr.attroff(curses.color_pair(4))
        except curses.error:
            try:
                stdscr.addstr(0, x, labels[: max(0, w - x)])
            except curses.error:
                stdscr.addstr(0, x, labels[: max(0, w - x)])

        # Status area (supports multi-line output). Use bold to simulate larger font.
        _render_status_lines(stdscr, (status or "").split("\n"))

        # Help footer
        help_text = "← → navigate  •  Enter select  •  q or ESC to quit"
        try:
            stdscr.attron(curses.A_DIM)
            stdscr.addstr(h - 1, 1, help_text[: max(0, w - 2)])
            stdscr.attroff(curses.A_DIM)
        except curses.error:
            stdscr.addstr(h - 1, 1, help_text[: max(0, w - 2)])

        stdscr.refresh()

    def _run_with_loading(stdscr, message: str, fn) -> Any:
        """Run a potentially long-running function in a thread while showing a
        fixed loading notice above the help footer: "Loading {year} Data...Please Wait !!".

        Returns the function's result (any type), or an error string on failure.
        """
        # Try to ensure color pair for red text on black exists; ignore failures on no-color terms
        _init_colors_if_needed()

        result: Dict[str, Any] = {"val": None}

        def _worker():
            try:
                out = fn()
                result["val"] = out
            except Exception as exc:  # pragma: no cover
                result["val"] = f"Error: {exc}"

        th = threading.Thread(target=_worker, daemon=True)
        th.start()

        # Make getch non-blocking while we animate the footer
        try:
            prev_nodelay = stdscr.nodelay(True)
        except Exception:
            prev_nodelay = False

        try:
            while th.is_alive():
                h, w = stdscr.getmaxyx()
                try:
                    # Draw help on bottom line remains; we draw loading on the line above it
                    # Clear the loading line first
                    y = max(0, h - 2)
                    stdscr.addstr(y, 0, " " * max(0, w))
                    text = (message or "").strip()
                    if text:
                        if hasattr(curses, "has_colors") and curses.has_colors():
                            stdscr.attron(curses.color_pair(8))
                        stdscr.addstr(y, 1, text[: max(0, w - 2)])
                        if hasattr(curses, "has_colors") and curses.has_colors():
                            stdscr.attroff(curses.color_pair(8))
                    stdscr.refresh()
                except Exception:
                    pass

                # Eat resize events so the footer stays at the bottom
                try:
                    ch = stdscr.getch()
                    if ch == curses.KEY_RESIZE:
                        # will recompute h,w on next loop
                        pass
                except Exception:
                    pass

                time.sleep(0.05)
        finally:
            try:
                stdscr.nodelay(prev_nodelay)
            except Exception:
                pass
        # Clear the loading line once done
        try:
            h, w = stdscr.getmaxyx()
            y = max(0, h - 2)
            stdscr.addstr(y, 0, " " * max(0, w))
            stdscr.refresh()
        except Exception:
            pass

        th.join(timeout=0.1)
        return result["val"] if result["val"] is not None else ""

    def _render_status_lines(
        stdscr,
        lines: List[str],
        highlight_idx: Optional[int] = None,
        scroll: int = 0,
        highlight_span: Optional[Tuple[int, int]] = None,  # (start, length) within the line
    ) -> None:
        """Render content lines starting at row 2.

        - highlight_idx: index in `lines` to highlight; if `highlight_span` is provided, only the
          given substring of that line is reversed instead of the entire row.
        - scroll: number of initial lines skipped for vertical scrolling.
        """
        h, w = stdscr.getmaxyx()
        max_rows = max(0, h - 3)
        start = max(0, scroll)
        end = min(len(lines), start + max_rows)
        visible = lines[start:end]
        for i, line in enumerate(visible):
            y = 2 + i
            status_line = line[: max(0, w - 2)]
            try:
                if has_colors:
                    stdscr.attron(curses.color_pair(3))
                stdscr.attron(curses.A_BOLD)
                # If this is the highlighted row and a span is provided, draw in 3 parts
                if highlight_idx is not None and (start + i) == highlight_idx and highlight_span:
                    span_start, span_len = highlight_span
                    span_start = max(0, span_start)
                    span_len = max(0, span_len)
                    # left segment
                    left = status_line[:span_start]
                    # span segment (reverse)
                    mid = status_line[span_start: span_start + span_len]
                    # right segment
                    right = status_line[span_start + span_len:]
                    x = 1
                    if left:
                        stdscr.addstr(y, x, left)
                        x += len(left)
                    if mid:
                        stdscr.attron(curses.A_REVERSE)
                        stdscr.addstr(y, x, mid)
                        stdscr.attroff(curses.A_REVERSE)
                        x += len(mid)
                    if right:
                        stdscr.addstr(y, x, right)
                else:
                    # Default: whole row optional reverse
                    if highlight_idx is not None and (start + i) == highlight_idx:
                        stdscr.attron(curses.A_REVERSE)
                    stdscr.addstr(y, 1, status_line)
                    if highlight_idx is not None and (start + i) == highlight_idx:
                        stdscr.attroff(curses.A_REVERSE)
                stdscr.attroff(curses.A_BOLD)
                if has_colors:
                    stdscr.attroff(curses.color_pair(3))
            except curses.error:
                try:
                    stdscr.attron(curses.A_BOLD)
                    if highlight_idx is not None and (start + i) == highlight_idx and highlight_span:
                        span_start, span_len = highlight_span
                        span_start = max(0, span_start)
                        span_len = max(0, span_len)
                        left = status_line[:span_start]
                        mid = status_line[span_start: span_start + span_len]
                        right = status_line[span_start + span_len:]
                        x = 1
                        if left:
                            stdscr.addstr(y, x, left); x += len(left)
                        if mid:
                            stdscr.attron(curses.A_REVERSE)
                            stdscr.addstr(y, x, mid)
                            stdscr.attroff(curses.A_REVERSE)
                            x += len(mid)
                        if right:
                            stdscr.addstr(y, x, right)
                    else:
                        if highlight_idx is not None and (start + i) == highlight_idx:
                            stdscr.attron(curses.A_REVERSE)
                        stdscr.addstr(y, 1, status_line)
                        if highlight_idx is not None and (start + i) == highlight_idx:
                            stdscr.attroff(curses.A_REVERSE)
                    stdscr.attroff(curses.A_BOLD)
                except curses.error:
                    pass

    def draw_with_lines(
        stdscr,
        selected_idx: int,
        lines: List[str],
        highlight_idx: Optional[int] = None,
        scroll: int = 0,
        help_text_override: Optional[str] = None,
        highlight_span: Optional[Tuple[int, int]] = None,
        *,
        focused: bool = True,
    ) -> None:
        # Redraw header and footer, then render provided lines with optional highlight
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        # Ensure colors and pairs are initialized consistently in this path too
        _init_colors_if_needed()

        # Top bar
        top_bar = " " * w
        try:
            if has_colors:
                if focused:
                    stdscr.attron(curses.color_pair(1))
                else:
                    if _inactive_pair_bg is not None:
                        stdscr.attron(curses.color_pair(9))
                    else:
                        stdscr.attron(curses.color_pair(1)); stdscr.attron(curses.A_DIM)
            stdscr.addstr(0, 0, top_bar)
            if has_colors:
                if focused:
                    stdscr.attroff(curses.color_pair(1))
                else:
                    if _inactive_pair_bg is not None:
                        stdscr.attroff(curses.color_pair(9))
                    else:
                        stdscr.attroff(curses.A_DIM); stdscr.attroff(curses.color_pair(1))
        except curses.error:
            stdscr.addstr(0, 0, top_bar)

        # Menu labels
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
                if focused:
                    stdscr.attron(curses.color_pair(4))
                else:
                    if _inactive_pair_bg is not None:
                        stdscr.attron(curses.color_pair(9))
                    else:
                        stdscr.attron(curses.color_pair(4)); stdscr.attron(curses.A_DIM)
            stdscr.attron(curses.A_BOLD)
            stdscr.addstr(0, x, labels[: max(0, w - x)])
            stdscr.attroff(curses.A_BOLD)
            if has_colors:
                if focused:
                    stdscr.attroff(curses.color_pair(4))
                else:
                    if _inactive_pair_bg is not None:
                        stdscr.attroff(curses.color_pair(9))
                    else:
                        stdscr.attroff(curses.A_DIM); stdscr.attroff(curses.color_pair(4))
        except curses.error:
            try:
                stdscr.attron(curses.A_BOLD)
                stdscr.addstr(0, x, labels[: max(0, w - x)])
                stdscr.attroff(curses.A_BOLD)
            except curses.error:
                stdscr.addstr(0, x, labels[: max(0, w - x)])

        # Content
        _render_status_lines(
            stdscr,
            lines,
            highlight_idx=highlight_idx,
            scroll=scroll,
            highlight_span=highlight_span,
        )

        # Help footer
        help_text = (
            help_text_override
            if help_text_override is not None
            else "← → navigate  •  Enter select  •  q or ESC to quit"
        )
        try:
            stdscr.attron(curses.A_DIM)
            stdscr.addstr(h - 1, 1, help_text[: max(0, w - 2)])
            stdscr.attroff(curses.A_DIM)
        except curses.error:
            stdscr.addstr(h - 1, 1, help_text[: max(0, w - 2)])

        stdscr.refresh()

    def _popup(stdscr, title: str, body_lines: List[str]) -> None:
        """Show a centered popup with a title and body lines. Close on Enter/ESC."""
        h, w = stdscr.getmaxyx()
        box_w = min(w - 6, max(len(title) + 4, max((len(l) for l in body_lines), default=0) + 4))
        box_h = min(h - 6, 4 + len(body_lines))
        start_y = max(2, (h - box_h) // 2)
        start_x = max(3, (w - box_w) // 2)
        win = stdscr.derwin(box_h, box_w, start_y, start_x)
        win.keypad(True)
        try:
            win.erase(); win.box()
            win.addnstr(0, max(1, (box_w - len(title)) // 2), title, box_w - 2)
            for i, line in enumerate(body_lines[: box_h - 2]):
                win.addnstr(1 + i, 2, line[: box_w - 4], box_w - 4)
            win.noutrefresh(); stdscr.noutrefresh(); curses.doupdate()
        except Exception:
            pass
        while True:
            ch = stdscr.getch()
            if ch in (27, curses.KEY_ENTER, 10, 13):
                break
            if ch == curses.KEY_RESIZE:
                # Close on resize and let parent redraw
                break

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
        curr_year = _dt.date.today().year
        next_year = curr_year + 1
        base = curr_year if base_year is None else int(base_year)
        # Always compute relative to actual current year for consistent UI behavior
        prev_years = [curr_year - i for i in range(count_previous)]
        if min_year is not None:
            prev_years = [y for y in prev_years if y >= int(min_year)]
        years = ([next_year] if include_next_year else []) + prev_years
        labels_display = ([show_next_label] if include_next_year else []) + [
            str(y) for y in prev_years
        ]
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
                win.addnstr(
                    0, max(1, (list_width - len(title)) // 2), title, list_width - 2
                )
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
                            win.addnstr(
                                1 + i, 1, label.ljust(list_width - 2), list_width - 2
                            )
                            win.attroff(curses.A_BOLD)
                            win.attroff(curses.color_pair(5))
                        else:
                            # Unselected: use dim attribute to simulate darker background effect
                            win.attron(curses.A_DIM)
                            win.addnstr(
                                1 + i, 1, label.ljust(list_width - 2), list_width - 2
                            )
                            win.attroff(curses.A_DIM)
                    else:
                        # No color support
                        marker = ">" if i == sel_idx else " "
                        win.addnstr(
                            1 + i,
                            1,
                            f"{marker} {disp}".ljust(list_width - 2),
                            list_width - 2,
                        )
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
            if ch in (27, ord("q")):  # ESC or q
                return None
            elif ch in (curses.KEY_UP, ord("k")):
                sel_idx = (sel_idx - 1) % len(years)
                _draw_dropdown()
            elif ch in (curses.KEY_DOWN, ord("j")):
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
        status = (
            "Welcome to BoxBox CLI.\nA Formula 1 data tracker for the command line."
        )
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
                        # Indicate menu loses focus while dropdown is visible
                        draw(stdscr, selected, status, focused=False)
                        chosen = select_year(stdscr, None)
                        if chosen is not None:
                            context["season"] = chosen
                        # After selection (or cancel), render calendar with current context
                        status = action(context) or ""
                    elif label == "Drivers":
                        # Show prior years but not before 2018; no next year entry
                        # Indicate menu loses focus while dropdown is visible
                        draw(stdscr, selected, status, focused=False)
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
                        # Show loading while fetching; drivers() may return dict with metadata
                        year = context.get("season")
                        msg = f"Loading {year} Data...Please Wait !!"
                        result = _run_with_loading(stdscr, msg, lambda: action(context))
                        from . import menu_actions as acts
                        # If error string, just show it
                        if isinstance(result, str):
                            status = result
                        else:
                            data = result or {}
                            lines = data.get("lines") or []
                            selectables = data.get("selectables") or []
                            season = data.get("season") or context.get("season")
                            # Enter selection mode if we have selectable rows
                            if lines and selectables:
                                sel_idx = 0
                                scroll = 0
                                # Precompute mapping of rows
                                row_to_idxs: dict[int, list[int]] = {}
                                for i, it in enumerate(selectables):
                                    r = int(it.get("row", 0))
                                    row_to_idxs.setdefault(r, []).append(i)
                                # Pre-warm driver enrichment in the background to speed up first popup
                                try:
                                    season_int = int(season or 0)
                                    driver_abbrs = sorted({
                                        str(it.get("abbr"))
                                        for it in selectables
                                        if it.get("type") == "driver" and it.get("abbr")
                                    })
                                    if season_int >= 2018 and driver_abbrs:
                                        th_prewarm = threading.Thread(
                                            target=lambda: acts.prewarm_driver_enrichment(season_int, driver_abbrs),
                                            daemon=True,
                                        )
                                        th_prewarm.start()
                                        # Also start a background completion to fill summaries for the whole season
                                        th_complete = threading.Thread(
                                            target=lambda: acts.complete_season_summaries(season_int),
                                            daemon=True,
                                        )
                                        th_complete.start()
                                except Exception:
                                    pass
                                # Main selection loop
                                # Column specification for cell highlighting
                                colspec = data.get("colspec") or {}
                                team_x = int(colspec.get("team_x", 0))
                                team_w = int(colspec.get("team_w", 0))
                                drv_x = int(colspec.get("driver_x", team_x + 2 + team_w))
                                drv_w = int(colspec.get("driver_w", 0))

                                # Build quick lookup maps for horizontal switching between
                                # Team and Driver even when they are not on the same row
                                team_to_team_idx: dict[str, int] = {}
                                team_to_first_driver_idx: dict[str, int] = {}
                                team_indices: List[int] = []
                                driver_indices: List[int] = []
                                for i2, it2 in enumerate(selectables):
                                    tname2 = it2.get("team")
                                    if isinstance(tname2, str) and tname2:
                                        if it2.get("type") == "team" and tname2 not in team_to_team_idx:
                                            team_to_team_idx[tname2] = i2
                                            team_indices.append(i2)
                                        elif it2.get("type") == "driver":
                                            driver_indices.append(i2)
                                            if tname2 not in team_to_first_driver_idx:
                                                team_to_first_driver_idx[tname2] = i2

                                while True:
                                    if not selectables:
                                        break
                                    item_for_span = selectables[sel_idx]
                                    hi_row = item_for_span.get("row", 0)
                                    # Determine span for partial highlight (only selected cell)
                                    span: Optional[Tuple[int, int]] = None
                                    itype = item_for_span.get("type")
                                    if isinstance(hi_row, int):
                                        if itype == "team" and team_w > 0:
                                            span = (team_x, team_w)
                                        elif itype == "driver" and drv_w > 0:
                                            span = (drv_x, drv_w)
                                    draw_with_lines(
                                        stdscr,
                                        selected,
                                        lines,
                                        highlight_idx=hi_row,
                                        scroll=scroll,
                                        help_text_override="↑ ↓ move  •  ← → switch Team/Driver  •  Enter details  •  ESC back",
                                        highlight_span=span,
                                        focused=False,
                                    )
                                    ch2 = stdscr.getch()
                                    if ch2 in (27, curses.KEY_BACKSPACE):
                                        # Exit selection mode back to drivers view
                                        status = "\n".join(lines)
                                        break
                                    elif ch2 in (curses.KEY_UP, ord('k')):
                                        cur = selectables[sel_idx]
                                        if cur.get("type") == "team" and team_indices:
                                            # move to previous team item
                                            try:
                                                pos = team_indices.index(sel_idx)
                                                sel_idx = team_indices[(pos - 1) % len(team_indices)]
                                            except ValueError:
                                                sel_idx = (sel_idx - 1) % len(selectables)
                                        elif cur.get("type") == "driver" and driver_indices:
                                            # move to previous driver item (stay in Driver column)
                                            try:
                                                pos = driver_indices.index(sel_idx)
                                                sel_idx = driver_indices[(pos - 1) % len(driver_indices)]
                                            except ValueError:
                                                sel_idx = (sel_idx - 1) % len(selectables)
                                        else:
                                            sel_idx = (sel_idx - 1) % len(selectables)
                                        # Ensure highlighted line stays in view
                                        hi_row2 = int(selectables[sel_idx].get("row", 0))
                                        h, w = stdscr.getmaxyx()
                                        max_rows = max(0, h - 3)
                                        if hi_row2 < scroll:
                                            scroll = hi_row2
                                        elif hi_row2 >= scroll + max_rows:
                                            scroll = hi_row2 - max_rows + 1
                                    elif ch2 in (curses.KEY_DOWN, ord('j')):
                                        cur = selectables[sel_idx]
                                        if cur.get("type") == "team" and team_indices:
                                            # move to next team item
                                            try:
                                                pos = team_indices.index(sel_idx)
                                                sel_idx = team_indices[(pos + 1) % len(team_indices)]
                                            except ValueError:
                                                sel_idx = (sel_idx + 1) % len(selectables)
                                        elif cur.get("type") == "driver" and driver_indices:
                                            # move to next driver item (stay in Driver column)
                                            try:
                                                pos = driver_indices.index(sel_idx)
                                                sel_idx = driver_indices[(pos + 1) % len(driver_indices)]
                                            except ValueError:
                                                sel_idx = (sel_idx + 1) % len(selectables)
                                        else:
                                            sel_idx = (sel_idx + 1) % len(selectables)
                                        h, w = stdscr.getmaxyx()
                                        max_rows = max(0, h - 3)
                                        hi_row2 = int(selectables[sel_idx].get("row", 0))
                                        if hi_row2 < scroll:
                                            scroll = hi_row2
                                        elif hi_row2 >= scroll + max_rows:
                                            scroll = hi_row2 - max_rows + 1
                                    elif ch2 in (curses.KEY_RIGHT, ord('l')):
                                        # Prefer switching within the same team
                                        cur = selectables[sel_idx]
                                        tname = cur.get("team")
                                        moved = False
                                        if isinstance(tname, str) and tname:
                                            if cur.get("type") == "team":
                                                nxt = team_to_first_driver_idx.get(tname)
                                                if isinstance(nxt, int):
                                                    sel_idx = nxt
                                                    moved = True
                                            elif cur.get("type") == "driver":
                                                nxt = team_to_team_idx.get(tname)
                                                if isinstance(nxt, int):
                                                    sel_idx = nxt
                                                    moved = True
                                        if not moved:
                                            # Fallback: toggle within same row if multiple items present
                                            row = int(cur.get("row", 0))
                                            options = row_to_idxs.get(row, [])
                                            if len(options) > 1:
                                                try:
                                                    pos = options.index(sel_idx)
                                                    sel_idx = options[(pos + 1) % len(options)]
                                                except ValueError:
                                                    pass
                                        # Keep selected row visible
                                        hi_row2 = int(selectables[sel_idx].get("row", 0))
                                        h, w = stdscr.getmaxyx()
                                        max_rows = max(0, h - 3)
                                        if hi_row2 < scroll:
                                            scroll = hi_row2
                                        elif hi_row2 >= scroll + max_rows:
                                            scroll = hi_row2 - max_rows + 1
                                    elif ch2 in (curses.KEY_LEFT, ord('h')):
                                        # Same as RIGHT, but reverse preference
                                        cur = selectables[sel_idx]
                                        tname = cur.get("team")
                                        moved = False
                                        if isinstance(tname, str) and tname:
                                            if cur.get("type") == "team":
                                                nxt = team_to_first_driver_idx.get(tname)
                                                if isinstance(nxt, int):
                                                    sel_idx = nxt
                                                    moved = True
                                            elif cur.get("type") == "driver":
                                                nxt = team_to_team_idx.get(tname)
                                                if isinstance(nxt, int):
                                                    sel_idx = nxt
                                                    moved = True
                                        if not moved:
                                            row = int(cur.get("row", 0))
                                            options = row_to_idxs.get(row, [])
                                            if len(options) > 1:
                                                try:
                                                    pos = options.index(sel_idx)
                                                    sel_idx = options[(pos - 1) % len(options)]
                                                except ValueError:
                                                    pass
                                        hi_row2 = int(selectables[sel_idx].get("row", 0))
                                        h, w = stdscr.getmaxyx()
                                        max_rows = max(0, h - 3)
                                        if hi_row2 < scroll:
                                            scroll = hi_row2
                                        elif hi_row2 >= scroll + max_rows:
                                            scroll = hi_row2 - max_rows + 1
                                    elif ch2 in (curses.KEY_NPAGE,):  # Page Down
                                        h, w = stdscr.getmaxyx()
                                        max_rows = max(1, h - 3)
                                        # move selection roughly one page
                                        sel_idx = min(len(selectables) - 1, sel_idx + max_rows)
                                        scroll += max_rows
                                    elif ch2 in (curses.KEY_PPAGE,):  # Page Up
                                        h, w = stdscr.getmaxyx()
                                        max_rows = max(1, h - 3)
                                        sel_idx = max(0, sel_idx - max_rows)
                                        scroll = max(0, scroll - max_rows)
                                    elif ch2 in (curses.KEY_HOME,):
                                        sel_idx = 0; scroll = 0
                                    elif ch2 in (curses.KEY_END,):
                                        sel_idx = len(selectables) - 1
                                        h, w = stdscr.getmaxyx()
                                        max_rows = max(0, h - 3)
                                        last_row = selectables[-1].get("row", 0)
                                        scroll = max(0, last_row - max_rows + 1)
                                    elif ch2 in (curses.KEY_ENTER, 10, 13):
                                        item = selectables[sel_idx]
                                        itype = item.get("type")
                                        if itype == "driver":
                                            abbr = item.get("abbr")
                                            y = int(season or 0)
                                            msg2 = f"Loading {y} Data...Please Wait !!"
                                            stext = _run_with_loading(
                                                stdscr,
                                                msg2,
                                                lambda: acts.driver_stats(context, y, abbr),
                                            )
                                            # Show popup
                                            body = str(stext or "").split("\n")
                                            _popup(stdscr, f"Driver {abbr}", body)
                                        elif itype == "team":
                                            team = item.get("team")
                                            y = int(season or 0)
                                            msg2 = f"Loading {y} Data...Please Wait !!"
                                            stext = _run_with_loading(
                                                stdscr,
                                                msg2,
                                                lambda: acts.constructor_stats(context, y, team),
                                            )
                                            body = str(stext or "").split("\n")
                                            _popup(stdscr, f"Constructor", body)
                                        # After popup, continue selection view
                                    elif ch2 == curses.KEY_RESIZE:
                                        # Re-draw with current scroll
                                        pass
                                # end selection loop
                            else:
                                status = "\n".join(lines) if lines else (result if isinstance(result, str) else "")
                    elif label == "Results":
                        # Same selector as Drivers (2018+ only, previous 20 years)
                        # Indicate menu loses focus while dropdown is visible
                        draw(stdscr, selected, status, focused=False)
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
                        # Show blinking loading footer while computing
                        year = context.get("season")
                        msg = f"Loading {year} Data...Please Wait !!"
                        status = _run_with_loading(stdscr, msg, lambda: action(context)) or ""
                    else:
                        status = action(context) or ""
                except Exception as e:  # pragma: no cover
                    status = f"Error: {e}"
            elif ch == curses.KEY_RESIZE:
                # Redraw on resize
                pass

            draw(stdscr, selected, status)

    curses.wrapper(main)
