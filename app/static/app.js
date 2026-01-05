document.getElementById("defectForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const summary = document.getElementById("Summary").value;
    const response = await fetch("/predict", {
        method: "POST",
        body: new URLSearchParams({ Summary: summary })
    });

    const result = await response.json();
    document.getElementById("result").innerHTML = `
        <p><strong>Status:</strong> ${result.Status}</p>
        <p><strong>Confidence:</strong> ${result.Confidence}</p>
        <p><strong>Valid %:</strong> ${result.Valid_Percentage}</p>
        <p><strong>Invalid %:</strong> ${result.Invalid_Percentage}</p>
    `;

    // Update confidence progress bar
    const confidenceValue = parseFloat(result.Confidence.replace("%", ""));
    const bar = document.getElementById("confidenceBar");
    bar.style.width = confidenceValue + "%";
});