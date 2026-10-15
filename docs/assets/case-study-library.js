window.COWORLD_CASE_STUDIES = [
  {
    id: "MMSI-Bench/MMSI-0003",
    slug: "mmsi-0003",
    title: "View change",
    operation: "MMSI-Bench",
    question: "When you took the second photo, where was the toilet in relation to you? Options: A: back right, B: front right, C: front left, D: back left",
    answer: "D: back left"
  },
  {
    id: "Spin-Bench/SPIN-2511",
    slug: "spin-2511",
    title: "Interpolation",
    operation: "Spin-Bench",
    question: "From a stationary viewpoint, in which direction did the object rotate between these two views? A. clockwise, B. counterclockwise. Only answer with the capital letter from (A, B). The camera is stationary and the object rotates in place. Clockwise and counterclockwise are defined from a top-down view.",
    answer: "B: counterclockwise"
  },
  {
    id: "SpatiaLab/SPATIALAB-590",
    slug: "spatialab-590",
    title: "Intervention",
    operation: "SpatiaLab",
    question: "If you take away the bottom plate from the stack, what is likely to happen? Choose exactly one option: A. The other plates will fall, B. The plates will float, C. Nothing will happen, D. All plates will break.",
    answer: "C: Nothing will happen"
  },
  {
    id: "CausalSpatial/CS-COLL-6169db902026",
    slug: "cs-coll-6169db902026",
    title: "Continuation",
    operation: "CausalSpatial",
    question: "A toy armed SUV with a rugged and tactical appearance lies on the floor. The toy car is driving forward. Is there any risk if it moves forward? A. Yes; B. No; C. Not Sure.",
    answer: "B: No"
  },
  {
    id: "CausalSpatial/CS-COLL-6a01e625175f",
    slug: "cs-coll-6a01e625175f",
    title: "Intervention",
    operation: "CausalSpatial",
    question: "On the floor, there is a toy truck with a classic design. If the toy car is driving forward, is it true that removing the white vase with a bold, oval-like form will prevent all collisions? If removing it does not prevent all collisions, what additional steps should be taken? A. Remove the pile of books; B. Remove the vase with a twisted loop; C. Remove the moon vase with gray colors; D. There is no collision; E. Removing the white vase with a bold, oval-like form is sufficient; F. Not Sure.",
    answer: "E. Removing the white vase with a bold, oval-like form is sufficient."
  }
];

document.addEventListener("DOMContentLoaded", () => {
  const root = document.getElementById("case-study-gallery");
  const cases = window.COWORLD_CASE_STUDIES;
  if (!root || !cases || !cases.length) return;

  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
  }[character]));
  const pageSize = 2;
  const pageCount = Math.ceil(cases.length / pageSize);
  let currentPage = 0;

  const card = (item) => {
    const base = `./assets/case-studies/${item.slug}`;
    return `<article class="case-card">
      <div class="case-card-head"><h4>${escapeHtml(item.title)}</h4><span>${escapeHtml(item.operation)} · ${escapeHtml(item.id.split("/").pop())}</span></div>
      <div class="case-card-question"><strong>Question</strong><p>${escapeHtml(item.question)}</p></div>
      <div class="case-card-media">
        <figure><img src="${base}/inputs.jpg" alt="Input images for ${escapeHtml(item.id)}" loading="lazy"><figcaption>Input</figcaption></figure>
        <figure><video controls preload="metadata" muted loop playsinline poster="${base}/poster.jpg"><source src="${base}/generated.mp4" type="video/mp4">Your browser does not support HTML video.</video><figcaption>Generated video</figcaption></figure>
      </div>
      <div class="case-card-answer"><strong>Answer</strong><span>${escapeHtml(item.answer)}</span></div>
      <div class="case-card-links"><a href="${base}/overview.jpg">Overview</a></div>
    </article>`;
  };

  const pages = Array.from({length: pageCount}, (_, page) => {
    const start = page * pageSize;
    return `<div class="case-page${page === 0 ? " is-active" : ""}" data-page="${page}">${cases.slice(start, start + pageSize).map(card).join("")}</div>`;
  }).join("");
  const controls = `<div class="case-pagination" role="navigation" aria-label="Example pages">
    <button type="button" class="case-page-prev" aria-label="Previous example page">Previous</button>
    <span class="case-page-status" aria-live="polite">Page 1 of ${pageCount}</span>
    <button type="button" class="case-page-next" aria-label="Next example page">Next</button>
  </div>`;
  root.innerHTML = `${pages}${controls}`;

  const pageElements = Array.from(root.querySelectorAll(".case-page"));
  const status = root.querySelector(".case-page-status");
  const previous = root.querySelector(".case-page-prev");
  const next = root.querySelector(".case-page-next");
  const alignActiveQuestions = () => {
    const questions = Array.from(pageElements[currentPage].querySelectorAll(".case-card-question"));
    questions.forEach((question) => { question.style.minHeight = ""; });
    if (window.matchMedia("(max-width: 600px)").matches || questions.length < 2) return;
    const maxHeight = Math.max(...questions.map((question) => question.getBoundingClientRect().height));
    questions.forEach((question) => { question.style.minHeight = `${maxHeight}px`; });
  };
  const update = () => {
    pageElements.forEach((page, index) => page.classList.toggle("is-active", index === currentPage));
    status.textContent = `Page ${currentPage + 1} of ${pageCount}`;
    previous.disabled = currentPage === 0;
    next.disabled = currentPage === pageCount - 1;
    requestAnimationFrame(alignActiveQuestions);
  };
  previous.addEventListener("click", () => { if (currentPage > 0) { currentPage -= 1; update(); } });
  next.addEventListener("click", () => { if (currentPage < pageCount - 1) { currentPage += 1; update(); } });
  window.addEventListener("resize", alignActiveQuestions);
  requestAnimationFrame(alignActiveQuestions);
});
