document.getElementById('runForm').addEventListener('submit', function(e) {
  e.preventDefault();

  document.getElementById('output').textContent = "";
  document.getElementById('error').textContent = "";

  const nFiles = document.getElementById('n_files').value;
  const trials = document.getElementById('trials').value;

  const streamUrl = `/stream?n_files=${encodeURIComponent(nFiles)}&trials=${encodeURIComponent(trials)}`;

  const evtSource = new EventSource(streamUrl);

  evtSource.onmessage = function(event) {
    document.getElementById('output').textContent += event.data;
  };

  evtSource.addEventListener('close', function(event) {
    console.log("Stream has closed successfully.");
    evtSource.close();
  });

  evtSource.onerror = function(err) {
    document.getElementById('error').textContent = "An error occurred while streaming output.";
    evtSource.close();
  };

  evtSource.onopen = function() {
    console.log("Connection to stream opened.");
  };
});