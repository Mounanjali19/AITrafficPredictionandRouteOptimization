document.getElementById("generate-map-btn").addEventListener("click", async () => {
    const start = parseInt(document.getElementById("map-start").value);
    const end = parseInt(document.getElementById("map-end").value);

    const res = await fetch("/api/route_map_full", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ start, end })
    });

    const data = await res.json();

    if (data.error) {
        alert("Error: " + data.error);
        return;
    }

    document.getElementById("map-frame").src = data.map_url;
});
