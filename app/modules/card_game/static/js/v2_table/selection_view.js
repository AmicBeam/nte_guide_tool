(function (root) {
  const TABLE = root.NTE_V2_TABLE = root.NTE_V2_TABLE || {};

  function bindSelectionView(ctx) {
    const button = ctx.nodes.selectionToggle;

    function restore(region) {
      if (!region) return;
      region.classList.remove('is-selection-collapsed');
      region.inert = false;
      region.setAttribute('aria-hidden', region === ctx.choiceOverlay && !region.classList.contains('open') ? 'true' : 'false');
    }

    function paint() {
      const view = ctx.selectionView;
      button.hidden = !view;
      ctx.page.classList.toggle('selection-collapsed', Boolean(view && view.collapsed));
      if (!view) return;
      view.region.classList.toggle('is-selection-collapsed', view.collapsed);
      view.region.inert = view.collapsed;
      view.region.setAttribute('aria-hidden', String(view.collapsed));
      button.setAttribute('aria-controls', view.region.id);
      button.setAttribute('aria-expanded', String(!view.collapsed));
      button.textContent = view.collapsed ? '返回' + view.label : '收起' + view.label + ' · 查看战场';
    }

    ctx.setSelectionView = function (config) {
      const previous = ctx.selectionView;
      if (!previous || previous.key !== config.key || previous.region !== config.region) {
        if (previous) restore(previous.region);
        ctx.selectionView = Object.assign({ collapsed: false }, config);
      }
      paint();
    };

    ctx.clearSelectionView = function (region) {
      if (!ctx.selectionView || (region && ctx.selectionView.region !== region)) return;
      restore(ctx.selectionView.region);
      ctx.selectionView = null;
      paint();
    };

    button.addEventListener('click', function () {
      const view = ctx.selectionView;
      if (!view) return;
      if (ctx.hideHoverPreview) ctx.hideHoverPreview({ force: true });
      view.collapsed = !view.collapsed;
      paint();
      if (view.collapsed) button.focus({ preventScroll: true });
    });
    paint();
  }

  TABLE.bindSelectionView = bindSelectionView;
}(window));
