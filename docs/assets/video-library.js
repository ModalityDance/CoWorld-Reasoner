window.COWORLD_VIDEOS = [
  {
    id: "SPIN-0036",
    category: "View change",
    source: "./assets/videos/043_SPIN-0036.mp4",
    poster: "./assets/video-posters/043_SPIN-0036.jpg",
    overview: "./assets/video-overviews/043_SPIN-0036.png",
    inputs: [
      "./assets/video-inputs/spin-0036/00_00_cars_rotation_e440cdbe67.jpg",
      "./assets/video-inputs/spin-0036/02_02_cars_rotation_a99108252e.jpg"
    ]
  },
  {
    id: "MMSI-0004",
    category: "Interpolation",
    source: "./assets/videos/007_MMSI-0004.mp4",
    poster: "./assets/video-posters/007_MMSI-0004.jpg",
    overview: "./assets/video-overviews/007_MMSI-0004.png",
    inputs: [
      "./assets/video-inputs/mmsi-0004/00_frame_1.jpg",
      "./assets/video-inputs/mmsi-0004/01_frame_2.jpg"
    ]
  },
  {
    id: "CS-COMP-3850AF8D19BE",
    category: "Continuation",
    source: "./assets/videos/029_CS-COMP-3850AF8D19BE.mp4",
    poster: "./assets/video-posters/029_CS-COMP-3850AF8D19BE.jpg",
    overview: "./assets/video-overviews/029_CS-COMP-3850AF8D19BE.png",
    inputs: ["./assets/video-inputs/cs-comp-3850af8d19be/00_image.png"]
  },
  {
    id: "CS-OCCL-8833CFBDF3AC",
    category: "Intervention",
    source: "./assets/videos/042_CS-OCCL-8833CFBDF3AC.mp4",
    poster: "./assets/video-posters/042_CS-OCCL-8833CFBDF3AC.jpg",
    overview: "./assets/video-overviews/042_CS-OCCL-8833CFBDF3AC.png",
    inputs: ["./assets/video-inputs/cs-occl-8833cfbdf3ac/00_image.png"]
  },
  {
    id: "MMSI-0034",
    category: "Intervention",
    source: "./assets/videos/047_MMSI-0034.mp4",
    poster: "./assets/video-posters/047_MMSI-0034.jpg",
    overview: "./assets/video-overviews/047_MMSI-0034.png",
    inputs: [
      "./assets/video-inputs/mmsi-0034/00_frame_1.jpg",
      "./assets/video-inputs/mmsi-0034/01_frame_2.jpg"
    ]
  },
  {
    id: "MMSI-0040",
    category: "View change",
    source: "./assets/videos/011_MMSI-0040.mp4",
    poster: "./assets/video-posters/011_MMSI-0040.jpg",
    overview: "./assets/video-overviews/011_MMSI-0040.png",
    inputs: [
      "./assets/video-inputs/mmsi-0040/00_frame_1.jpg",
      "./assets/video-inputs/mmsi-0040/01_frame_2.jpg"
    ]
  },
  {
    id: "MMSI-0010",
    category: "View change",
    source: "./assets/videos/002_MMSI-0010.mp4",
    poster: "./assets/video-posters/002_MMSI-0010.jpg",
    overview: "./assets/video-overviews/002_MMSI-0010.png",
    inputs: [
      "./assets/video-inputs/mmsi-0010/00_frame_1.jpg",
      "./assets/video-inputs/mmsi-0010/01_frame_2.jpg"
    ]
  },
  {
    id: "MMSI-0041",
    category: "View change",
    source: "./assets/videos/058_MMSI-0041.mp4",
    poster: "./assets/video-posters/058_MMSI-0041.jpg",
    overview: "./assets/video-overviews/058_MMSI-0041.png",
    inputs: [
      "./assets/video-inputs/mmsi-0041/00_frame_1.jpg",
      "./assets/video-inputs/mmsi-0041/01_frame_2.jpg"
    ]
  },
  {
    id: "CS-OCCL-F21E9C7F1C66",
    category: "Intervention",
    source: "./assets/videos/001_CS-OCCL-F21E9C7F1C66.mp4",
    poster: "./assets/video-posters/001_CS-OCCL-F21E9C7F1C66.jpg",
    overview: "./assets/video-overviews/001_CS-OCCL-F21E9C7F1C66.png",
    inputs: ["./assets/video-inputs/cs-occl-f21e9c7f1c66/00_image.png"]
  },
  {
    id: "MMSI-0048",
    category: "Interpolation",
    source: "./assets/videos/048_MMSI-0048.mp4",
    poster: "./assets/video-posters/048_MMSI-0048.jpg",
    overview: "./assets/video-overviews/048_MMSI-0048.png",
    inputs: [
      "./assets/video-inputs/mmsi-0048/01.jpg",
      "./assets/video-inputs/mmsi-0048/02.jpg"
    ]
  }
];

document.addEventListener("DOMContentLoaded", () => {
  const root = document.getElementById("video-gallery");
  const videos = window.COWORLD_VIDEOS;
  if (!root || !videos || !videos.length) return;

  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
  }[character]));
  const pageSize = 2;
  const pageCount = Math.ceil(videos.length / pageSize);
  let currentPage = 0;

  const card = (item) => {
    const inputs = item.inputs.map((src, index) =>
      `<img src="${src}" alt="Input ${index + 1} for ${escapeHtml(item.id)}" loading="lazy">`
    ).join("");
    return `<figure class="video-card">
      <div class="video-card-input-label">Input</div>
      <div class="video-inputs" data-count="${item.inputs.length}">${inputs}</div>
      <div class="video-card-media-label">Generated video</div>
      <div class="video-frame"><video controls preload="metadata" muted loop playsinline poster="${item.poster}" aria-label="${escapeHtml(item.id)}"><source src="${item.source}" type="video/mp4">Your browser does not support HTML video.</video></div>
      <figcaption><div class="video-card-head"><span class="video-card-title">${escapeHtml(item.id)}</span><span class="video-card-tag">${escapeHtml(item.category)}</span></div></figcaption>
    </figure>`;
  };

  const pages = Array.from({length: pageCount}, (_, page) => {
    const start = page * pageSize;
    return `<div class="video-page${page === 0 ? " is-active" : ""}" data-page="${page}">${videos.slice(start, start + pageSize).map(card).join("")}</div>`;
  }).join("");
  const controls = `<div class="video-pagination" role="navigation" aria-label="Video pages">
    <button type="button" class="video-page-prev" aria-label="Previous video page">Previous</button>
    <span class="video-page-status" aria-live="polite">Page 1 of ${pageCount}</span>
    <button type="button" class="video-page-next" aria-label="Next video page">Next</button>
  </div>`;
  root.innerHTML = `${pages}${controls}`;

  const pageElements = Array.from(root.querySelectorAll(".video-page"));
  const status = root.querySelector(".video-page-status");
  const previous = root.querySelector(".video-page-prev");
  const next = root.querySelector(".video-page-next");
  const update = () => {
    pageElements.forEach((page, index) => page.classList.toggle("is-active", index === currentPage));
    status.textContent = `Page ${currentPage + 1} of ${pageCount}`;
    previous.disabled = currentPage === 0;
    next.disabled = currentPage === pageCount - 1;
  };
  previous.addEventListener("click", () => { if (currentPage > 0) { currentPage -= 1; update(); } });
  next.addEventListener("click", () => { if (currentPage < pageCount - 1) { currentPage += 1; update(); } });
});
