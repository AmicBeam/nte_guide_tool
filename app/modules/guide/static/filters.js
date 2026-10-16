"use strict";
document.querySelectorAll("[data-guide-filters]").forEach(filters => {
  const form = filters.closest("form");
  const cards = Array.from(document.querySelectorAll("[data-filter-card]"));
  const count = document.querySelector("[data-guide-result-count]");
  const empty = document.querySelector("[data-guide-empty]");
  function refresh() {
    const values = new FormData(form);
    const query = String(values.get("q") || "").trim().toLocaleLowerCase();
    let matches = 0;
    cards.forEach(card => {
      const shown = (!values.get("element") || card.dataset.element === values.get("element")) &&
        (!values.get("rarity") || card.dataset.rarity === values.get("rarity")) &&
        (!values.get("arc_type") || card.dataset.arcType === values.get("arc_type")) &&
        (!query || card.dataset.filterText.includes(query));
      card.hidden = !shown;
      if (shown) matches += 1;
    });
    document.querySelectorAll("[data-guide-filter-group]").forEach(group => {
      const visible = group.querySelectorAll('[data-filter-card]:not([hidden])').length;
      group.hidden = visible === 0;
      group.querySelector("[data-guide-group-count]").textContent = `${visible} 个弧盘`;
    });
    count.textContent = `${matches} 条结果`;
    empty.hidden = matches > 0;
    const url = new URL(location.href);
    ["q", "element", "rarity", "arc_type"].forEach(key => {
      const value = String(values.get(key) || "").trim();
      if (value) url.searchParams.set(key, value);
      else url.searchParams.delete(key);
    });
    history.replaceState(null, "", url);
  }
  empty.querySelector("a").addEventListener("click", event => {
    event.preventDefault();
    form.querySelector('input[type="search"]').value = "";
    filters.querySelectorAll('input[type="radio"]').forEach(input => { input.checked = input.value === ""; });
    refresh();
  });
  form.addEventListener("submit", event => { event.preventDefault(); refresh(); });
  filters.addEventListener("change", refresh);
  form.querySelector('input[type="search"]').addEventListener("input", refresh);
});
