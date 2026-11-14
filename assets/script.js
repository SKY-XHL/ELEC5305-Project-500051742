(() => {
  const links = window.PROJECT_LINKS || {};
  const configuredLinks = document.querySelectorAll("[data-project-link]");

  configuredLinks.forEach((link) => {
    const key = link.dataset.projectLink;
    const url = typeof links[key] === "string" ? links[key].trim() : "";
    if (!url) return;
    link.href = url;
    link.removeAttribute("aria-disabled");
    if (key === "video") link.textContent = "Watch presentation ↗";
    if (key === "repository") link.textContent = "Open GitHub repository ↗";
    if (/^https?:/i.test(url)) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
  });

  const configuredCount = [...configuredLinks].filter(
    (link) => link.getAttribute("aria-disabled") !== "true"
  ).length;
  const status = document.querySelector("#link-status");
  if (status && configuredCount === configuredLinks.length) status.hidden = true;

  const toggle = document.querySelector(".nav-toggle");
  const navLinks = document.querySelector("#nav-links");
  if (toggle && navLinks) {
    toggle.addEventListener("click", () => {
      const open = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!open));
      navLinks.classList.toggle("open", !open);
    });
    navLinks.addEventListener("click", (event) => {
      if (!event.target.closest("a")) return;
      toggle.setAttribute("aria-expanded", "false");
      navLinks.classList.remove("open");
    });
  }

  document.querySelectorAll('a[href="#top"]').forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
      history.replaceState(null, "", `${location.pathname}${location.search}#top`);
    });
  });

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const reveals = document.querySelectorAll(".reveal");
  if (reduceMotion || !("IntersectionObserver" in window)) {
    reveals.forEach((item) => item.classList.add("visible"));
  } else {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -30px" });
    reveals.forEach((item) => observer.observe(item));
  }

  const filterButtons = document.querySelectorAll(".filter-button");
  const galleryCards = document.querySelectorAll(".gallery-card");
  filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const selected = button.dataset.filter;
      filterButtons.forEach((item) => {
        const active = item === button;
        item.classList.toggle("active", active);
        item.setAttribute("aria-pressed", String(active));
      });
      galleryCards.forEach((card) => {
        const categories = (card.dataset.category || "").split(/\s+/);
        card.classList.toggle("is-hidden", selected !== "all" && !categories.includes(selected));
      });
    });
  });

  const lightbox = document.querySelector("#lightbox");
  const lightboxImage = lightbox?.querySelector("img");
  const lightboxCaption = lightbox?.querySelector("p");
  const lightboxClose = lightbox?.querySelector(".lightbox-close");

  const closeLightbox = () => {
    if (lightbox?.open) lightbox.close();
  };

  document.querySelectorAll(".figure-open").forEach((button) => {
    button.addEventListener("click", () => {
      if (!lightbox || !lightboxImage || !lightboxCaption) return;
      lightboxImage.src = button.dataset.image || button.querySelector("img")?.src || "";
      lightboxImage.alt = button.dataset.alt || button.querySelector("img")?.alt || "Project figure";
      lightboxCaption.textContent = lightboxImage.alt;
      document.body.classList.add("modal-open");
      lightbox.showModal();
      lightboxClose?.focus();
    });
  });

  lightboxClose?.addEventListener("click", closeLightbox);
  lightbox?.addEventListener("click", (event) => {
    if (event.target === lightbox) closeLightbox();
  });
  lightbox?.addEventListener("close", () => {
    document.body.classList.remove("modal-open");
    if (lightboxImage) lightboxImage.src = "";
  });
})();
