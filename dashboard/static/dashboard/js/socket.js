(function () {
  const listeners = {};

  function on(eventName, handler) {
    (listeners[eventName] = listeners[eventName] || []).push(handler);
  }

  function dispatch(eventName, payload) {
    (listeners[eventName] || []).forEach((fn) => fn(payload));
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/training/`);

  ws.onopen = () => dispatch('_open', {});
  ws.onclose = () => dispatch('_close', {});
  ws.onmessage = (evt) => {
    let data;
    try {
      data = JSON.parse(evt.data);
    } catch (e) {
      return;
    }
    dispatch(data.event, data.payload);
  };

  function send(action, extra) {
    if (ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(Object.assign({ action }, extra || {})));
  }

  window.DashboardSocket = { on, send };
})();
