// Light/dark theme for both demo pages. Starts from the OS setting; clicking
// the header's theme button switches it and remembers the choice in
// localStorage. Loaded in <head> (not deferred) so the theme is applied
// before first paint, with no flash of the wrong colors.

(function () {
  const STORAGE_KEY = "voicecart-theme";

  function savedTheme() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (err) { return null; }
  }

  function apply(theme) {
    document.documentElement.dataset.theme = theme;
    const button = document.getElementById("theme-toggle");
    if (button) button.setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
  }

  const osDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  apply(savedTheme() || (osDark ? "dark" : "light"));

  document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("theme-toggle");
    apply(document.documentElement.dataset.theme);
    button.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      apply(next);
      // Storage can be unavailable (private window, blocked site data);
      // the switch still works for this page view.
      try { localStorage.setItem(STORAGE_KEY, next); } catch (err) { /* ignore */ }
    });
  });
})();
