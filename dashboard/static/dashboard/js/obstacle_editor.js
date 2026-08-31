(function () {
  const COLS = window.DASHBOARD_DATA.cols;
  const ROWS = window.DASHBOARD_DATA.rows;

  // Must stay in sync with env.py's reset(): head = [cols//2, rows//2],
  // body extends 2 cells to the left of the head.
  const spawnHead = [Math.floor(COLS / 2), Math.floor(ROWS / 2)];
  const spawnCells = new Set([
    `${spawnHead[0]},${spawnHead[1]}`,
    `${spawnHead[0] - 1},${spawnHead[1]}`,
    `${spawnHead[0] - 2},${spawnHead[1]}`,
  ]);

  const grid = document.getElementById('obstacle-grid');
  grid.style.gridTemplateColumns = `repeat(${COLS}, 1fr)`;

  const obstacles = new Set();
  const cellEls = {};

  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const key = `${x},${y}`;
      const cell = document.createElement('div');
      cell.className = 'obs-cell';
      if (spawnCells.has(key)) {
        cell.classList.add('spawn');
      } else {
        cell.addEventListener('click', () => toggleCell(key, cell));
      }
      cellEls[key] = cell;
      grid.appendChild(cell);
    }
  }

  function toggleCell(key, cell) {
    if (obstacles.has(key)) {
      obstacles.delete(key);
      cell.classList.remove('obstacle');
    } else {
      obstacles.add(key);
      cell.classList.add('obstacle');
    }
  }

  function clearAll() {
    obstacles.forEach((key) => cellEls[key] && cellEls[key].classList.remove('obstacle'));
    obstacles.clear();
  }

  function setCells(cells) {
    clearAll();
    (cells || []).forEach(([x, y]) => {
      const key = `${x},${y}`;
      if (spawnCells.has(key) || !cellEls[key]) return;
      obstacles.add(key);
      cellEls[key].classList.add('obstacle');
    });
  }

  function getCells() {
    return Array.from(obstacles).map((key) => key.split(',').map(Number));
  }

  window.DashboardObstacleEditor = { clearAll, setCells, getCells };
})();
