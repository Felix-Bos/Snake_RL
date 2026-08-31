(function () {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { labels: { color: '#8891a7', boxWidth: 10, font: { size: 10 } } } },
    scales: {
      x: { ticks: { color: '#8891a7', font: { size: 9 } }, grid: { color: '#1b1f2b' } },
      y: { ticks: { color: '#8891a7', font: { size: 9 } }, grid: { color: '#1b1f2b' } },
    },
  };

  const rewardCtx = document.getElementById('chart-reward').getContext('2d');
  const lossCtx = document.getElementById('chart-loss').getContext('2d');

  const rewardChart = new Chart(rewardCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Score', data: [], borderColor: '#4ade80', backgroundColor: 'transparent', pointRadius: 0, borderWidth: 1.5 },
        { label: 'Moy. 50', data: [], borderColor: '#60a5fa', backgroundColor: 'transparent', pointRadius: 0, borderWidth: 2 },
      ],
    },
    options: commonOptions,
  });

  const lossChart = new Chart(lossCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Loss', data: [], borderColor: '#f87171', backgroundColor: 'transparent', pointRadius: 0, borderWidth: 1.5 },
      ],
    },
    options: commonOptions,
  });

  const MAX_POINTS = 300;

  function pushPoint(chart, label, values) {
    chart.data.labels.push(label);
    chart.data.datasets.forEach((ds, i) => ds.data.push(values[i]));
    if (chart.data.labels.length > MAX_POINTS) {
      chart.data.labels.shift();
      chart.data.datasets.forEach((ds) => ds.data.shift());
    }
    chart.update('none');
  }

  function addMetrics(metrics) {
    pushPoint(rewardChart, metrics.episode, [metrics.score, metrics.mean_score_50]);
    if (metrics.loss !== null && metrics.loss !== undefined) {
      pushPoint(lossChart, metrics.episode, [metrics.loss]);
    }
  }

  function reset() {
    [rewardChart, lossChart].forEach((chart) => {
      chart.data.labels = [];
      chart.data.datasets.forEach((ds) => (ds.data = []));
      chart.update('none');
    });
  }

  window.DashboardCharts = { addMetrics, reset };
})();
