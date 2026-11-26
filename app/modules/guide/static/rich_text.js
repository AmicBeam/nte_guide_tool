'use strict';
(() => {
    let active = null;
    let dismissed = null;
    let closeTimer;
    const hide = () => {
        clearTimeout(closeTimer);
        if (!active) return;
        active.tip.hidden = true;
        active.button.setAttribute('aria-expanded', 'false');
        active = null;
    };
    const show = button => {
        clearTimeout(closeTimer);
        if (active?.button === button) return;
        hide();
        const tip = document.getElementById(button.getAttribute('aria-describedby'));
        if (!tip) return;
        tip.hidden = false;
        button.setAttribute('aria-expanded', 'true');
        active = {button, tip};
        const rect = button.getBoundingClientRect();
        const bounds = tip.getBoundingClientRect();
        const left = Math.max(16, Math.min(rect.left, window.innerWidth - bounds.width - 16));
        const below = rect.bottom + 10;
        const top = below + bounds.height <= window.innerHeight - 16 ? below : Math.max(16, rect.top - bounds.height - 10);
        tip.style.left = `${left}px`;
        tip.style.top = `${top}px`;
    };
    document.addEventListener('pointerover', event => {
        const button = event.target.closest('.guide-term');
        if (button && button !== dismissed && event.pointerType !== 'touch') show(button);
        if (active?.tip.contains(event.target)) clearTimeout(closeTimer);
    });
    document.addEventListener('pointerout', event => {
        if (dismissed?.contains(event.target) && !dismissed.contains(event.relatedTarget)) dismissed = null;
        if (!active || (!active.button.contains(event.target) && !active.tip.contains(event.target))) return;
        if (active.button.contains(event.relatedTarget) || active.tip.contains(event.relatedTarget)) return;
        closeTimer = setTimeout(hide, 180);
    });
    document.addEventListener('focusin', event => {
        if (event.target.matches('.guide-term')) { dismissed = null; show(event.target); }
        else if (active && !active.tip.contains(event.target)) hide();
    });
    document.addEventListener('click', event => {
        const button = event.target.closest('.guide-term');
        if (button) { dismissed = null; show(button); }
        else if (active && !active.tip.contains(event.target)) hide();
    });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape') { dismissed = active?.button; hide(); }
    });
    window.addEventListener('resize', hide);
    document.addEventListener('scroll', event => { if (active && !active.tip.contains(event.target)) hide(); }, true);
})();
