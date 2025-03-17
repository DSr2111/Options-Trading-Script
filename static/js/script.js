document.getElementById('input-form').addEventListener('submit', function () {
  const submitBtn = document.getElementById('submit-btn');
  const btnText = document.getElementById('btn-text');
  const spinner = document.getElementById('spinner');

  submitBtn.disabled = true;
  btnText.textContent = 'Processing...';
  spinner.classList.remove('d-none');
});

// Apply conditional formatting after the table is loaded
document.addEventListener('DOMContentLoaded', function () {
  const table = document.getElementById('results-table');
  if (table) {
    const rows = table.querySelectorAll('tbody tr');
    const popValues = [];

    // Collect all PoP values
    rows.forEach((row) => {
      const popCell = row.cells[10]; // PoP is the 11th column (index 10)
      const popValue = parseFloat(popCell.textContent);
      popValues.push(popValue);
    });

    // Sort PoP values and find the threshold for top 25%
    popValues.sort((a, b) => b - a);
    const thresholdIndex = Math.floor(popValues.length * 0.25);
    const popThreshold = popValues[thresholdIndex] || popValues[0];

    // Apply high-pop class to rows with PoP in top 25%
    rows.forEach((row) => {
      const popCell = row.cells[10];
      const popValue = parseFloat(popCell.textContent);
      if (popValue >= popThreshold) {
        row.classList.add('high-pop');
      }
    });
  }
});
