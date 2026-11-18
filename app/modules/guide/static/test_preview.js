'use strict';
(() => {
  const page = document.querySelector('[data-guide-test-preview]');
  if (!page) return;
  const token = () => { try { return localStorage.getItem('nte_token') || ''; } catch (_) { return ''; } };
  const initialToken = token();
  if (page.dataset.guideTestPreview === 'true') {
    const recheck = () => { if (token() !== initialToken) window.location.reload(); };
    window.addEventListener('storage', recheck);
    window.addEventListener('pageshow', event => { if (event.persisted) window.location.reload(); });
    document.addEventListener('visibilitychange', recheck);
    return;
  }
  if (!initialToken) return;
  // Native page navigation cannot send the saved Bearer token. Fetch the same
  // read-only page with it, then render the server's authorized HTML and scripts.
  fetch(window.location.href, {headers: {Authorization: `Bearer ${initialToken}`}, cache: 'no-store', redirect: 'error'})
    .then(async response => {
      if (!response.ok || !response.headers.get('Content-Type')?.includes('text/html')) return;
      const html = await response.text();
      if (token() !== initialToken) return;
      const parsed = new DOMParser().parseFromString(html, 'text/html');
      if (parsed.querySelector('[data-guide-test-preview="true"]')) {
        document.open();
        document.write(html);
        document.close();
      }
    }).catch(() => { /* Keep the public page available if preview fails. */ });
})();
