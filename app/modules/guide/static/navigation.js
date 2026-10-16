'use strict';

(() => {
  const link = document.getElementById('guide-account-link');
  const loginUrl = () => '/login?next=' + encodeURIComponent(location.pathname + location.search + location.hash);
  let checking = null;
  const render = (token) => {
    if (link) {
      link.textContent = token ? '账户' : '登录';
      link.href = token ? '/profile' : loginUrl();
    }
    return token;
  };
  const check = () => {
    if (checking) return checking;
    checking = (async () => {
      let token = '';
      try { token = localStorage.getItem('nte_token') || ''; } catch (_) { /* anonymous browsing */ }
      if (!token) return render('');
      try {
        const response = await fetch('/api/me', { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' });
        if (response.status === 401) return render('');
      } catch (_) { /* temporary network failures do not erase a saved login */ }
      return render(token);
    })().finally(() => { checking = null; });
    return checking;
  };
  window.guideAuth = { check, loginUrl, ready: check() };
  window.addEventListener('storage', event => { if (event.key === 'nte_token') check(); });
  window.addEventListener('pageshow', event => { if (event.persisted) check(); });
})();
