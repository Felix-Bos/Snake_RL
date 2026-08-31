(function () {
  const canvas = document.getElementById('game-canvas');
  const ctx = canvas.getContext('2d');

  const COLORS = {
    grid: '#1b1f2b',
    obstacle: '#4b5262',
    body: '#22c55e',
    head: '#4ade80',
    food: '#f87171',
  };

  function drawRoundedCell(x, y, cellPx, color, glow) {
    const pad = 1;
    ctx.fillStyle = color;
    if (glow) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
    } else {
      ctx.shadowBlur = 0;
    }
    const size = cellPx - pad * 2;
    const r = Math.min(5, size / 2);
    const px = x * cellPx + pad;
    const py = y * cellPx + pad;
    ctx.beginPath();
    ctx.moveTo(px + r, py);
    ctx.arcTo(px + size, py, px + size, py + size, r);
    ctx.arcTo(px + size, py + size, px, py + size, r);
    ctx.arcTo(px, py + size, px, py, r);
    ctx.arcTo(px, py, px + size, py, r);
    ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  function drawGrid(cols, rows, cellPx) {
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= cols; x++) {
      ctx.moveTo(x * cellPx, 0);
      ctx.lineTo(x * cellPx, rows * cellPx);
    }
    for (let y = 0; y <= rows; y++) {
      ctx.moveTo(0, y * cellPx);
      ctx.lineTo(cols * cellPx, y * cellPx);
    }
    ctx.stroke();
  }

  function drawGameState(state) {
    const cellPx = canvas.width / state.cols;
    canvas.height = state.rows * cellPx;

    ctx.fillStyle = '#060709';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawGrid(state.cols, state.rows, cellPx);

    (state.obstacles || []).forEach(([x, y]) => drawRoundedCell(x, y, cellPx, COLORS.obstacle, false));
    (state.snake || []).slice(1).forEach(([x, y]) => drawRoundedCell(x, y, cellPx, COLORS.body, false));
    if (state.food) drawRoundedCell(state.food[0], state.food[1], cellPx, COLORS.food, true);
    if (state.head) drawRoundedCell(state.head[0], state.head[1], cellPx, COLORS.head, true);
  }

  window.DashboardCanvas = { drawGameState };
})();
