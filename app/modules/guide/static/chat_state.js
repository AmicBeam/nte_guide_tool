/* Browser-only history, partitioned by the authenticated website player. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.GuideChatState = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const GAP_MS = 6 * 60 * 60 * 1000;
  function expired(session, now = Date.now()) {
    const last = session?.messages?.at(-1);
    return Boolean(last && now - last.timestamp > GAP_MS);
  }
  function context(messages) {
    let pictures = 6, text = 60000;
    const rows = [];
    for (const message of [...messages].slice(-32).reverse()) {
      const images = message.images || [];
      const included = images.length <= pictures;
      const references = message.role === 'assistant' ? (message.sources || []).slice(0,20).map(source =>
        `${String(source.title).slice(0,160)}: ${String(source.url).slice(0,500)}`).join('\n').slice(0,4000) : '';
      const suffix = (references ? `\n[上轮回答引用的资料，仅供追问定位]\n${references}` : '') +
        (!included && images.length ? '\n[早前图片未包含，需要时请重发。]' : '');
      const bounded = message.text.slice(0, Math.max(0,12000-suffix.length)) + suffix;
      if (bounded.length > text) break;
      text -= bounded.length;
      rows.push({role:message.role, text:bounded,
        timestamp:message.timestamp, images:included ? images.map(image => typeof image === 'string' ? image : image.data) : []});
      if (included) pictures -= images.length;
    }
    return rows.reverse();
  }
  function open() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('nte-guide-chat-v1', 1);
      request.onupgradeneeded = () => {
        const store = request.result.createObjectStore('sessions', {keyPath:['user', 'id']});
        store.createIndex('user', 'user');
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(new Error('浏览器无法保存对话，请检查浏览器存储权限。'));
    });
  }
  function operation(db, mode, action) {
    return new Promise((resolve, reject) => {
      const transaction = db.transaction('sessions', mode);
      const result = action(transaction.objectStore('sessions'));
      transaction.oncomplete = () => resolve(result.result);
      transaction.onerror = () => reject(new Error('对话保存失败，可能是浏览器空间不足。'));
      transaction.onabort = transaction.onerror;
    });
  }
  return {GAP_MS, expired, context, open,
    list: (db,user) => operation(db,'readonly',store=>store.index('user').getAll(user)),
    save: (db,session) => operation(db,'readwrite',store=>store.put(session)),
    remove: (db,user,id) => operation(db,'readwrite',store=>store.delete([user,id]))};
}));
