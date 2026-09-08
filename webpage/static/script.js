const form = document.getElementById("convert-form");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const resultEl = document.getElementById("result");
const downloadLink = document.getElementById("download-link");
const errorEl = document.getElementById("error");

function setBusy(isBusy) {
  submitBtn.disabled = isBusy;
  statusEl.hidden = !isBusy;
  if (isBusy) {
    resultEl.hidden = true;
    errorEl.hidden = true;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(true);
  statusText.textContent = "Processing... this can take a few minutes if splitting audio.";

  const formData = new FormData(form);
  // Checkboxes only appear in FormData when checked; normalize to explicit true/false.
  formData.set("quantize", document.getElementById("quantize").checked ? "true" : "false");
  formData.set("split_audio", document.getElementById("split_audio").checked ? "true" : "false");

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed with status ${response.status}`);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    downloadLink.href = url;

    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="?([^"]+)"?/);
    downloadLink.download = match ? match[1] : "output.mid";

    resultEl.hidden = false;
  } catch (err) {
    errorEl.textContent = err.message || "Something went wrong.";
    errorEl.hidden = false;
  } finally {
    setBusy(false);
  }
});
