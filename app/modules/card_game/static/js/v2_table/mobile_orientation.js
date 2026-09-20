(function () {
  const page = document.getElementById('v2-table-page');
  const button = document.getElementById('v2-landscape-btn');
  const hint = document.getElementById('v2-rotate-copy');
  const orientation = window.screen && window.screen.orientation;
  if (!page || !window.matchMedia('(pointer: coarse)').matches
      || !orientation || typeof orientation.lock !== 'function') return;

  let attempted = false;
  let locked = false;
  let leaving = false;
  if (button) button.hidden = false;

  function unlock() {
    if (!locked) return;
    locked = false;
    try { orientation.unlock(); } catch (_) { /* Optional browser capability. */ }
  }

  async function tryLandscape(event) {
    // Fullscreen needs a gesture in this document; navigation from the lobby
    // cannot carry that permission over. Never intercept the actual game action.
    if (attempted || !event.isTrusted
        || event.target.closest('a, #v2-result-leave, #v2-home-link')) return;
    attempted = true;
    page.removeEventListener('click', tryLandscape, true);
    if (button) button.hidden = true;
    const root = document.documentElement;
    try {
      if (!document.fullscreenElement && typeof root.requestFullscreen === 'function') {
        await root.requestFullscreen();
      }
    } catch (_) { /* Standalone browsers may allow locking without fullscreen. */ }
    if (leaving) return;
    try {
      await orientation.lock('landscape');
      locked = true;
      if (leaving) unlock();
    } catch (_) {
      if (hint) hint.textContent = '请手动旋转手机后继续对战。';
    }
  }

  page.addEventListener('click', tryLandscape, { capture: true, passive: true });
  document.addEventListener('fullscreenchange', function () {
    if (!document.fullscreenElement) unlock();
  });
  window.addEventListener('pagehide', function () {
    leaving = true;
    unlock();
  });
  window.addEventListener('pageshow', function () { leaving = false; });
})();
