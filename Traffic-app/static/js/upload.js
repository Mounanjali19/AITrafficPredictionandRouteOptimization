// static/js/upload.js

document.addEventListener("DOMContentLoaded", () => {
    initUploadForm();
});

function initUploadForm() {
    const form = document.getElementById("uploadForm");
    const input = document.getElementById("uploadImage");
    const resultEl = document.getElementById("uploadResult");
    const previewOriginal = document.getElementById("previewOriginal");
    const previewDetected = document.getElementById("previewDetected");

    if (!form || !input || !resultEl || !previewOriginal || !previewDetected) return;

    // Show local preview of selected image
    input.addEventListener("change", () => {
        const file = input.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewOriginal.src = e.target.result;
            previewOriginal.style.display = "block";
        };
        reader.readAsDataURL(file);
    });

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        resultEl.textContent = "Uploading image and running YOLOv8 detection...";

        const file = input.files[0];
        if (!file) {
            resultEl.textContent = "❌ Please choose an image first.";
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        try {
            const res = await fetch("/api/yolo_detect", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const txt = await res.text();
                throw new Error(`HTTP ${res.status}: ${txt}`);
            }

            const data = await res.json();
            if (data.error) throw new Error(data.error);

            const count = data.vehicle_count;
            const annotatedUrl = data.annotated_image_url;

            let text = "✅ YOLOv8 Detection Complete\n\n";
            if (count !== undefined) {
                text += `Detected vehicles: ${count}\n`;
            }

            if (annotatedUrl) {
                previewDetected.src = annotatedUrl;
                previewDetected.style.display = "block";
                text += "\nShowing annotated image on the right.";
            }

            resultEl.textContent = text;
        } catch (err) {
            console.error(err);
            resultEl.textContent = "❌ Error: " + formatError(err);
        }
    });
}
