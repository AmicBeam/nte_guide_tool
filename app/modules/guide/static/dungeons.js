"use strict";

document.querySelector('.dungeon-stage-form select')?.addEventListener('change', event => {
  event.currentTarget.form.requestSubmit();
});
