// static/js/influence.js

document.addEventListener("DOMContentLoaded", () => {
    initInfluence();
});

function initInfluence() {
    const btn = document.getElementById("loadInfluence");
    const statusEl = document.getElementById("influenceStatus");
    const container = document.getElementById("heatmapContainer");

    if (!btn || !statusEl || !container) return;

    btn.addEventListener("click", async () => {
        statusEl.textContent = "Fetching influence / attention matrix...";
        container.innerHTML = "";

        try {
            const res = await fetch("/api/influence");
            if (!res.ok) {
                const txt = await res.text();
                throw new Error(`HTTP ${res.status}: ${txt}`);
            }

            const data = await res.json();
            if (data.error) throw new Error(data.error);

            const roads = data.roads || [];
            const matrix = data.matrix || [];

            if (!roads.length || !matrix.length) {
                statusEl.textContent = "No influence data returned.";
                return;
            }

            // Compute min/max for color scaling
            let min = Infinity;
            let max = -Infinity;
            matrix.forEach(row => {
                row.forEach(v => {
                    if (typeof v === "number") {
                        if (v < min) min = v;
                        if (v > max) max = v;
                    }
                });
            });
            if (!isFinite(min) || !isFinite(max)) {
                min = 0;
                max = 1;
            }
            const range = max - min || 1;

            const table = document.createElement("table");

            // Header row
            const thead = document.createElement("thead");
            const headRow = document.createElement("tr");

            const emptyTh = document.createElement("th");
            emptyTh.textContent = "";
            headRow.appendChild(emptyTh);

            roads.forEach(r => {
                const th = document.createElement("th");
                th.textContent = r;
                headRow.appendChild(th);
            });

            thead.appendChild(headRow);
            table.appendChild(thead);

            // Body
            const tbody = document.createElement("tbody");
            matrix.forEach((row, i) => {
                const tr = document.createElement("tr");

                const rowHeader = document.createElement("th");
                rowHeader.textContent = roads[i] || `R${i}`;
                tr.appendChild(rowHeader);

                row.forEach((val) => {
                    const td = document.createElement("td");
                    const v = typeof val === "number" ? val : 0;
                    const norm = (v - min) / range;
                    const intensity = Math.max(0.1, norm);
                    td.style.backgroundColor = `rgba(59, 130, 246, ${0.2 + intensity * 0.6})`;
                    td.textContent = v.toFixed ? v.toFixed(2) : v;
                    tr.appendChild(td);
                });

                tbody.appendChild(tr);
            });

            table.appendChild(tbody);
            container.appendChild(table);

            statusEl.textContent = "✅ Influence matrix loaded.";
        } catch (err) {
            console.error(err);
            statusEl.textContent = "❌ Error: " + formatError(err);
        }
    });
}
