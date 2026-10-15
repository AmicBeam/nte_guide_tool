'use strict';
importScripts(`pair_engine.js${self.location.search}`);
self.onmessage = ({data}) => {
    try {
        const result = self.OverlapPairEngine.match(data.characters, data.options, progress => self.postMessage({progress}));
        self.postMessage({result});
    } catch (error) { self.postMessage({error: error.message}); }
};
