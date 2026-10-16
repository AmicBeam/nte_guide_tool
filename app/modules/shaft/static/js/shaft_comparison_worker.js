'use strict';
const version = new URL(self.location.href).searchParams.get('v') || '';
importScripts(`shaft_engine.js?v=${version}`, `shaft_self_check.js?v=${version}`, `shaft_comparison.js?v=${version}`);
self.onmessage = event => {
  try {
    const {axis, catalog, selection} = event.data;
    for (const message of ShaftComparison.compare(axis, catalog, selection, ShaftEngine, ShaftSelfCheck)) {
      self.postMessage(message);
    }
    self.postMessage({type: 'done'});
  } catch (error) {
    self.postMessage({type: 'error', error: error.message || '比较失败，请减少组合后重试。'});
  }
};
