document
  .getElementById('input-form')
  .addEventListener('submit', function (event) {
    event.preventDefault();

    const portfolioValue = parseFloat(
      document.getElementById('portfolio_value').value
    );
    const minPop = parseFloat(document.getElementById('min_pop').value);
    const targetPremium = parseFloat(
      document.getElementById('target_premium').value
    );
    const minDays = parseInt(document.getElementById('min_days').value);
    const maxDays = parseInt(document.getElementById('max_days').value);
    const errorMessage = document.getElementById('error-message');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('spinner');

    // Reset error message
    errorMessage.classList.add('d-none');
    errorMessage.textContent = '';

    // Validation checks
    let errors = [];
    if (isNaN(portfolioValue) || portfolioValue <= 0) {
      errors.push('Portfolio Value must be a positive number.');
    }
    if (isNaN(minPop) || minPop < 0 || minPop > 100) {
      errors.push('Minimum PoP must be between 0 and 100.');
    }
    if (isNaN(targetPremium) || targetPremium <= 0) {
      errors.push('Target Premium must be a positive number.');
    }
    if (isNaN(minDays) || minDays < 1) {
      errors.push('Min Days to Expiry must be at least 1.');
    }
    if (isNaN(maxDays) || maxDays < minDays) {
      errors.push(
        'Max Days to Expiry must be greater than or equal to Min Days.'
      );
    }

    if (errors.length > 0) {
      errorMessage.textContent = errors.join(' ');
      errorMessage.classList.remove('d-none');
      return;
    }

    // If validation passes, proceed with form submission
    submitBtn.disabled = true;
    btnText.textContent = 'Processing...';
    spinner.classList.remove('d-none');
    this.submit();
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

    // Add CSV download functionality
    document
      .getElementById('download-csv')
      .addEventListener('click', function (e) {
        e.preventDefault();
        const headers = Array.from(table.querySelectorAll('thead th')).map(
          (th) => th.textContent
        );
        const rows = Array.from(table.querySelectorAll('tbody tr')).map((row) =>
          Array.from(row.cells)
            .map((cell) => `"${cell.textContent}"`)
            .join(',')
        );
        const csvContent = [headers.join(','), ...rows].join('\n');
        const blob = new Blob([csvContent], {
          type: 'text/csv;charset=utf-8;',
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', 'iron_condor_recommendations.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      });
  }
});
