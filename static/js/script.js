document.getElementById('input-form').addEventListener('submit', function () {
  const submitBtn = document.getElementById('submit-btn');
  const btnText = document.getElementById('btn-text');
  const spinner = document.getElementById('spinner');

  submitBtn.disabled = true;
  btnText.textContent = 'Processing...';
  spinner.classList.remove('d-none');
});
