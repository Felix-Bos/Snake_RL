(function () {
  const $ = (id) => document.getElementById(id);

  const statusPill = $('status-pill');
  const errorBanner = $('error-banner');

  const btnStartTrain = $('btn-start-train');
  const btnStop = $('btn-stop');
  const btnStartWatch = $('btn-start-watch');
  const btnStopWatch = $('btn-stop-watch');

  // --- Tabs ---
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      $(`tab-${btn.dataset.tab}`).classList.add('active');
    });
  });

  // --- Checkpoints dropdowns ---
  function populateCheckpoints(checkpoints) {
    const names = checkpoints.map((c) => c.name);
    [$('cfg-resume-model'), $('watch-model')].forEach((select) => {
      const current = select.value;
      select.innerHTML = '';
      names.forEach((name) => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      });
      if (names.includes(current)) select.value = current;
    });
  }

  populateCheckpoints(window.DASHBOARD_DATA.checkpoints || []);

  function refreshCheckpoints() {
    fetch('/api/checkpoints/').then((r) => r.json()).then((data) => populateCheckpoints(data.checkpoints));
  }

  $('cfg-resume').addEventListener('change', (e) => {
    $('resume-model-field').style.display = e.target.checked ? 'block' : 'none';
  });

  // --- Saved obstacle layouts dropdown ---
  function populateLayouts(layouts) {
    const select = $('saved-layouts-select');
    select.innerHTML = '<option value="">— Charger une carte —</option>';
    layouts.forEach((name) => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    });
  }
  populateLayouts(window.DASHBOARD_DATA.savedLayouts || []);

  $('btn-clear-obstacles').addEventListener('click', () => window.DashboardObstacleEditor.clearAll());

  $('btn-save-obstacles').addEventListener('click', () => {
    const name = prompt('Nom de la carte :');
    if (!name) return;
    fetch('/api/obstacle-layouts/save/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCsrfToken() },
      body: JSON.stringify({ name, cells: window.DashboardObstacleEditor.getCells() }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) { showError(data.error); return; }
        fetch('/api/obstacle-layouts/').then((r) => r.json()).then((d) => populateLayouts(d.layouts));
      });
  });

  $('btn-load-layout').addEventListener('click', () => {
    const name = $('saved-layouts-select').value;
    if (!name) return;
    fetch(`/api/obstacle-layouts/${encodeURIComponent(name)}/`)
      .then((r) => r.json())
      .then((data) => {
        if (data.error) { showError(data.error); return; }
        window.DashboardObstacleEditor.setCells(data.cells);
      });
  });

  function getCsrfToken() {
    const match = document.cookie.match(/csrftoken=([^;]+)/);
    return match ? match[1] : '';
  }

  // --- Error banner ---
  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.classList.add('visible');
  }
  function clearError() {
    errorBanner.classList.remove('visible');
    errorBanner.textContent = '';
  }

  // --- Status handling ---
  function applyStatus(status) {
    statusPill.textContent = status;
    statusPill.dataset.status = status;

    const running = status === 'training' || status === 'watching' || status === 'stopping';
    btnStartTrain.disabled = running;
    btnStartWatch.disabled = running;
    btnStop.disabled = !running;
    btnStopWatch.disabled = !running;
  }

  applyStatus('idle');

  // --- Start/Stop actions ---
  btnStartTrain.addEventListener('click', () => {
    clearError();
    DashboardCharts.reset();
    const config = {
      run_name: $('cfg-run-name').value.trim(),
      algo: $('cfg-algo').value,
      obs: $('cfg-obs').value,
      episodes: Number($('cfg-episodes').value),
      max_steps: Number($('cfg-max-steps').value),
      batch_size: Number($('cfg-batch-size').value),
      lr: Number($('cfg-lr').value),
      obstacles: Number($('cfg-obstacles').value),
      no_food_steps: Number($('cfg-no-food-steps').value),
      step_penalty: Number($('cfg-step-penalty').value),
      train_speed: $('cfg-train-speed').value,
      use_custom_obstacles: $('cfg-use-custom-obstacles').checked,
      resume: $('cfg-resume').checked,
      resume_model: $('cfg-resume-model').value || '',
    };
    const payload = { config };
    if (config.use_custom_obstacles) {
      config.obstacle_layout = window.DashboardObstacleEditor.getCells();
    }
    DashboardSocket.send('start_training', payload);
  });

  btnStartWatch.addEventListener('click', () => {
    clearError();
    const modelName = $('watch-model').value;
    if (!modelName) { showError('Aucun modèle disponible. Entraînez-en un dabord.'); return; }
    const useCustom = $('cfg-watch-use-custom-obstacles').checked;
    DashboardSocket.send('start_watch', {
      model_name: modelName,
      obs_type: $('watch-obs').value,
      n_episodes: Number($('watch-episodes').value),
      obstacle_layout: useCustom ? window.DashboardObstacleEditor.getCells() : null,
    });
  });

  btnStop.addEventListener('click', () => DashboardSocket.send('stop'));
  btnStopWatch.addEventListener('click', () => DashboardSocket.send('stop'));

  // --- WebSocket event wiring ---
  DashboardSocket.on('status_change', (payload) => {
    applyStatus(payload.status);
    if (payload.error) showError(payload.error);
    if (payload.status === 'idle') refreshCheckpoints();
  });

  DashboardSocket.on('metrics_update', (payload) => {
    $('stat-episode').textContent = payload.episode;
    $('stat-score').textContent = payload.score.toFixed(1);
    $('stat-mean').textContent = payload.mean_score_50.toFixed(1);
    $('stat-epsilon').textContent = payload.epsilon.toFixed(3);
    $('stat-loss').textContent = payload.loss !== null && payload.loss !== undefined ? payload.loss.toFixed(4) : '–';
    $('stat-buffer').textContent = payload.buffer_size;
    $('stat-time').textContent = `${Math.round(payload.elapsed_sec)}s`;
    DashboardCharts.addMetrics(payload);
  });

  DashboardSocket.on('game_state', (payload) => {
    DashboardCanvas.drawGameState(payload);
    $('stat-score').textContent = payload.score.toFixed ? payload.score.toFixed(1) : payload.score;
  });

  DashboardSocket.on('episode_finished', (payload) => {
    $('stat-episode').textContent = payload.episode;
    $('stat-score').textContent = payload.score.toFixed(1);
  });

  DashboardSocket.on('training_finished', (payload) => {
    if (payload.reason === 'error' && payload.message) showError(payload.message);
    refreshCheckpoints();
  });

  DashboardSocket.on('error', (payload) => showError(payload.message));
})();
