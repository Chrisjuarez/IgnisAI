const express   = require('express');
const router    = express.Router();
const { spawn } = require('child_process');
const path      = require('path');

router.post('/', (req, res) => {
  const fireData = req.body;
  if (!fireData?.lat || !fireData?.lng) {
    return res.status(400).json({ error: 'Missing required fire location data' });
  }

  // Path to your Python script
  const pythonScript = path.join(__dirname, '../ml/predict_spread.py');
  const venvDir      = path.join(__dirname, '../ml/venv');

  // Select the right interpreter for macOS/Linux vs. Windows
  const pythonPath = process.platform === 'win32'
    ? path.join(venvDir, 'Scripts', 'python.exe')
    : path.join(venvDir, 'bin',     'python3');

  console.log('Spawning Python at:', pythonPath);

  // Accumulators for stdout/stderr
  let stdout = '';
  let errorOutput = '';

  // Spawn the Python process
  const python = spawn(pythonPath, [
    pythonScript,
    JSON.stringify(fireData)
  ]);

  // Collect standard output
  python.stdout.on('data', data => {
    stdout += data.toString();
  });

  // Collect error output
  python.stderr.on('data', data => {
    errorOutput += data.toString();
  });

  // When the script exits...
  python.on('close', code => {
    if (code !== 0) {
      console.error('Python script failed:', errorOutput);
      return res.status(500).json({
        error:   'Fire spread prediction failed',
        details: errorOutput
      });
    }

    try {
      // Parse and return the JSON result
      const prediction = JSON.parse(stdout);
      res.json(prediction);
    } catch (e) {
      console.error('Failed to parse prediction results:', e);
      res.status(500).json({
        error:   'Failed to parse prediction results',
        details: e.message
      });
    }
  });
});

module.exports = router;