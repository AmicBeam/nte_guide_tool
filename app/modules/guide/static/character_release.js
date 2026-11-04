'use strict';
(() => {
  let refreshing = false;
  document.querySelectorAll('[data-character-release-url]').forEach(node => {
    let timer;
    let pending = false;
    const check = async () => {
      if (pending || refreshing) return;
      clearTimeout(timer);
      pending = true;
      let delay = 30000;
      try {
        const response = await fetch(node.dataset.characterReleaseUrl, {cache:'no-store'});
        if (response.ok) {
          const {data} = await response.json();
          if (data.content_released === true) {
            if (!refreshing) {
              refreshing = true;
              window.location.reload();
            }
            return;
          }
          if (Number.isFinite(data.check_after_ms)) delay = Math.max(250, Math.min(30000, data.check_after_ms));
        }
      } catch (_) { /* Retry without using the browser's clock as publication authority. */ }
      finally { pending = false; }
      if (!refreshing) timer = setTimeout(check, delay);
    };
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') check();
    });
    check();
  });
})();
