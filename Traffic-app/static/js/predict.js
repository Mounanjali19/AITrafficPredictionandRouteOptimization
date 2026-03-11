// -----------------------------
// HYBRID GAT-LSTM PREDICTION
// -----------------------------
document.getElementById("hybrid-btn").addEventListener("click", async () => {

    const date = document.getElementById("hybrid-date").value;
    const time = document.getElementById("time_input").value;   // TIME INCLUDED
    const scenario = document.getElementById("hybrid-scenario").value;

    if (!date) {
        alert("Please select a valid date.");
        return;
    }
    if (!time) {
        alert("Please select a valid time.");
        return;
    }

    const payload = {
        date: date,
        time: time,
        scenario: scenario
    };

    const res = await fetch("/api/hybrid_predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await res.json();
    const box = document.getElementById("hybrid-result");

    if (data.error) {
        box.textContent = "❌ Error: " + data.error;
        return;
    }

    // show results
    const roads = data.roads || [];
    const speeds = data.speeds || [];
    const unit = data.unit || "km/h";

    let lines = [];

    lines.push("✅ Hybrid Multi-Road Prediction\n");
    lines.push(`Scenario : ${data.scenario}`);
    lines.push(`Date     : ${data.date}`);
    lines.push(`Time     : ${data.time}\n`);
    lines.push(`Road Speed (${unit})`);
    lines.push("------------------------");

    for (let i = 0; i < roads.length; i++) {
        lines.push(`${roads[i]}  ${speeds[i].toFixed(2)}`);
    }

    box.textContent = lines.join("\n");
});

// -----------------------------
// PPO ROUTE PREDICTION (uses last hybrid result)
// -----------------------------
document.getElementById("ppo-btn").addEventListener("click", async () => {

    const start = document.getElementById("ppo-start").value;
    const end = document.getElementById("ppo-end").value;

    const date = document.getElementById("hybrid-date").value;
    const time = document.getElementById("time_input").value;    // TIME INCLUDED
    const scenario = document.getElementById("hybrid-scenario").value || "normal";

    if (start === "" || end === "") {
        alert("Enter both start and end edges.");
        return;
    }
    if (!date) {
        alert("Please select a valid date.");
        return;
    }
    if (!time) {
        alert("Please select a valid time.");
        return;
    }

    const payload = {
        start: parseInt(start),
        end: parseInt(end),
        date: date,
        time: time,
        scenario: scenario
    };

    const res = await fetch("/api/ppo_route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await res.json();
    const box = document.getElementById("ppo-result");

    if (data.error) {
        box.textContent = "❌ Error: " + data.error;
        return;
    }

    let lines = [];
    lines.push("🚦 PPO Route Recommendation\n");
    lines.push(`Start edge : R${data.start}`);
    lines.push(`End edge   : R${data.end}`);
    lines.push(`Date       : ${data.date}`);
    lines.push(`Time       : ${data.time}`);
    lines.push(`Scenario   : ${data.scenario}\n`);
    lines.push(`Recommended edge index: ${data.recommended_route_index}`);
    lines.push(`Predicted Speed       : ${data.predicted_speed.toFixed(2)} km/h\n`);
    lines.push(`Explanation: ${data.note}`);

    box.textContent = lines.join("\n");
});
