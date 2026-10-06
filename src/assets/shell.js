// Fullscreen for the page-shell canvas. Figures set a fixed layout height, so
// the plot is stretched to the screen on entry and restored on exit; Plotly
// only follows the container width by itself.
document.addEventListener("fullscreenchange", function () {
  const canvas = document.getElementById("shell-canvas");
  const gd = canvas && canvas.querySelector(".js-plotly-plot");
  if (!gd || !window.Plotly) {
    return;
  }
  if (document.fullscreenElement === canvas) {
    gd.dataset.shellHeight = gd.layout.height || "";
    window.Plotly.relayout(gd, { height: window.innerHeight - 48 });
  } else {
    const h = Number(gd.dataset.shellHeight);
    window.Plotly.relayout(gd, { height: h || null });
  }
});
