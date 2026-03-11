// static/js/app.js

// ------------------------------
// This file handles theme toggle
// ------------------------------

/**
 * Safe function wrapper so VS Code doesn't complain
 */
document.addEventListener("DOMContentLoaded", function () {
    setupThemeToggle();
});

/**
 * Change theme + store preference in localStorage
 */
function setupThemeToggle() {
    var body = document.body;
    var btn = document.getElementById("themeToggle");

    if (!btn) return; // prevent null errors

    // Load saved theme
    var saved = localStorage.getItem("traffic_theme");
    if (saved === "light" || saved === "dark") {
        body.setAttribute("data-theme", saved);
        btn.textContent = saved === "dark" ? "🌙" : "☀️";
    } else {
        body.setAttribute("data-theme", "dark");
        btn.textContent = "🌙";
    }

    // Toggle button click
    btn.addEventListener("click", function () {
        var current = body.getAttribute("data-theme") || "dark";
        var next = current === "dark" ? "light" : "dark";

        body.setAttribute("data-theme", next);
        btn.textContent = next === "dark" ? "🌙" : "☀️";

        localStorage.setItem("traffic_theme", next);
    });
}

/**
 * Global error formatter function
 */
function formatError(err) {
    if (!err) return "Unknown error";

    if (typeof err === "string") return err;

    if (err.message) return err.message;

    try {
        return JSON.stringify(err);
    } catch (e) {
        return String(err);
    }
}
