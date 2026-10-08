const V2 = window.NTE_V2;

if (V2.ensureLogin()) {
  bootstrapReplays();
}

async function bootstrapReplays() {
  const banner = document.getElementById('v2-replay-banner');
  const replayList = document.getElementById('v2-replay-list');
  const replayCopy = document.getElementById('v2-replay-copy');
  const importInput = document.getElementById('v2-replay-import');
  let busy = false;

  function replayStatus(item) {
    if (item.winner === 'draw') return '平局';
    if (item.winner === item.viewer_side) return '胜利';
    if (item.winner) return '失败';
    return item.status === 'playing' ? '进行中' : '已结束';
  }

  function lineupMarkup(characters) {
    const list = Array.isArray(characters) ? characters.slice(0, 4) : [];
    if (!list.length) {
      return '<span class="v2-replay-lineup is-empty" aria-hidden="true"></span>';
    }
    return '<span class="v2-replay-lineup">' + list.map(function (item) {
      const character = Object.assign({}, V2.characterFallback(item && item.id) || {}, item || {});
      const avatar = V2.characterAsset(character, 'avatar') || V2.characterAsset(character, 'portrait');
      return V2.imageMarkup(avatar, character.name, 'v2-replay-avatar');
    }).join('') + '</span>';
  }

  function renderReplayList(payload) {
    const items = (payload && payload.replays) || [];
    const limit = payload && payload.unfavorited_limit ? payload.unfavorited_limit : 30;
    replayCopy.textContent = '只展示你参加过的对局公开局面。可收藏；未收藏最多保留 ' + limit + ' 场。训练评估可用 --export-replays 转成 JSON 后再导入观看。';
    if (!items.length) {
      replayList.innerHTML = '<p class="subtle" id="v2-replay-empty">还没有可回放的对局。开始一局人机或 1v1 后即可在此查看。</p>';
      return;
    }
    replayList.innerHTML = items.map(function (item) {
      const mode = item.mode === 'advanced' ? '高级人机' : (item.mode === 'solo' ? '人机' : '1v1');
      const names = V2.escapeHtml((item.name_a || '我方') + ' vs ' + (item.name_b || '对手'));
      const starred = Boolean(item.favorited);
      const when = V2.formatReplayTime(item.updated_at);
      return (
        '<article class="v2-replay-row' + (starred ? ' is-starred' : '') + '">' +
          '<div class="v2-replay-lineups" aria-label="双方构筑">' +
            lineupMarkup(item.characters_a) +
            '<span class="v2-replay-vs">vs</span>' +
            lineupMarkup(item.characters_b) +
          '</div>' +
          '<div class="v2-replay-copy">' +
            '<h3>房间 ' + V2.escapeHtml(item.room_code) + ' · ' + mode + '</h3>' +
            '<p class="subtle">' + (when ? V2.escapeHtml(when) + ' · ' : '') + names + ' · ' + V2.escapeHtml(replayStatus(item)) +
            ' · ' + V2.escapeHtml(String(item.turn_count != null ? item.turn_count : item.event_count || 0)) + ' 个回合</p>' +
          '</div>' +
          '<div class="v2-replay-actions">' +
            '<button class="secondary-btn" type="button" data-star-replay="' + V2.escapeAttr(item.room_code) +
              '" data-starred="' + (starred ? '1' : '0') + '">' + (starred ? '已收藏' : '收藏') + '</button>' +
            '<a class="secondary-btn" href="/table?replay=' + V2.escapeAttr(item.room_code) + '">观看回放</a>' +
          '</div>' +
        '</article>'
      );
    }).join('');
  }

  if (importInput) {
    importInput.addEventListener('change', async function () {
      const file = importInput.files && importInput.files[0];
      importInput.value = '';
      if (!file || busy) {
        return;
      }
      busy = true;
      try {
        const payload = JSON.parse(await file.text());
        const imported = await V2.importReplay(payload);
        const code = imported && imported.replay && imported.replay.room_code;
        renderReplayList(await V2.listReplays());
        V2.showBanner(banner, code ? ('已导入，房间 ' + code) : '已导入训练录像。', 'ok');
      } catch (error) {
        V2.showBanner(banner, error && error.message ? error.message : '无法导入训练录像。', 'error');
      } finally {
        busy = false;
      }
    });
  }

  replayList.addEventListener('click', async function (event) {
    const button = event.target.closest('[data-star-replay]');
    if (!button || busy) {
      return;
    }
    const code = button.getAttribute('data-star-replay');
    const next = button.getAttribute('data-starred') !== '1';
    busy = true;
    button.disabled = true;
    try {
      renderReplayList(await V2.starReplay(code, next));
    } catch (error) {
      V2.showBanner(banner, error && error.message ? error.message : '无法更新收藏。', 'error');
    } finally {
      busy = false;
    }
  });

  try {
    renderReplayList(await V2.listReplays());
  } catch (error) {
    replayList.innerHTML = '<p class="subtle">暂时无法读取最近回放。</p>';
    if (!(error && error.status === 404)) {
      V2.showBanner(banner, error && error.message ? error.message : '无法读取回放列表。', 'error');
    }
  }
}
