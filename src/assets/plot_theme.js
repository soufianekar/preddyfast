// Plotly figures follow the theme, in the browser, with no server round trip.
//
// The server draws figures with transparent paper; this script sets text,
// grid, zero-line, legend and hover colours from the theme tokens
// (window.MX_THEMES, generated from components/theme_toggle.py) on every
// figure as it renders, and again on every theme switch. Data colours are
// never touched: they are chosen to read on both themes. Secondary text — the
// title's subtitle and the provenance footer (the annotation named
// "provenance", see figures._frame_timeseries) — takes the muted token.
(function () {
  function theme() {
    return document.documentElement.getAttribute("data-mx-theme") || "dark";
  }

  function mxPlotUpdate(layout, name) {
    const t = (window.MX_THEMES || {})[name] || {};
    const u = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      "font.color": t.ink,
      "title.font.color": t.ink,
      "legend.bgcolor": "rgba(0,0,0,0)",
      "legend.font.color": t.ink,
      "hoverlabel.bgcolor": t.raise,
      "hoverlabel.bordercolor": t.rule,
      "hoverlabel.font.color": t.ink,
    };
    if (layout && layout.title && layout.title.subtitle) {
      u["title.subtitle.font.color"] = t.dim;
    }
    ((layout && layout.annotations) || []).forEach(function (a, i) {
      if (a && a.name === "provenance") {
        u["annotations[" + i + "].font.color"] = t.dim;
      }
    });
    Object.keys(layout || {}).forEach(function (key) {
      if (/^[xy]axis\d*$/.test(key)) {
        u[key + ".gridcolor"] = t.grid;
        u[key + ".zerolinecolor"] = t.zero;
        u[key + ".linecolor"] = t.rule;
        u[key + ".tickfont.color"] = t.dim;
        u[key + ".title.font.color"] = t.dim;
      }
    });
    return u;
  }
  window.mxPlotUpdate = mxPlotUpdate;

  // Time-series frame (figures._frame_timeseries): the server places the
  // legend and provenance footer from an estimated legend height; here they
  // follow the legend's measured height, which depends on the screen width.
  // Returns the relayout update, or null when nothing needs to move.
  function mxFrameFit(full, layout) {
    const frame = layout && layout.meta && layout.meta.mx_frame;
    const i = ((layout && layout.annotations) || []).findIndex(function (a) {
      return a && a.name === "provenance";
    });
    if (!frame || i < 0 || !full || !full.legend) {
      return null;
    }
    const footerTop = frame.axis_px + (full.legend._height || 0) + frame.gap_px;
    const bottom = footerTop + frame.lines * frame.line_px + frame.pad_px;
    if (Math.abs(bottom - full.margin.b) < 2) {
      return null;
    }
    const u = { "margin.b": bottom };
    u["legend.y"] = -frame.axis_px / (full.height - full.margin.t - bottom);
    u["annotations[" + i + "].yshift"] = -footerTop;
    return u;
  }
  window.mxFrameFit = mxFrameFit;

  function fit(gd) {
    if (!window.Plotly || gd._mxExporting) {
      return;
    }
    const u = mxFrameFit(gd._fullLayout, gd.layout);
    if (u) {
      window.Plotly.relayout(gd, u);
    }
  }

  // A figure is themed once per (theme, layout object). dcc.Graph swaps in a
  // new layout object for every new figure (Plotly.react), which drops the
  // earlier restyle; relayout mutates the same object, so it does not loop.
  function apply(gd) {
    const name = theme();
    if (!window.Plotly || !gd.layout || gd._mxExporting) {
      return;
    }
    if (gd._mxTheme === name && gd._mxLayout === gd.layout) {
      return;
    }
    gd._mxTheme = name; // set first: relayout fires plotly_afterplot again
    gd._mxLayout = gd.layout;
    window.Plotly.relayout(gd, mxPlotUpdate(gd.layout, name));
  }

  function watch(gd) {
    if (gd._mxWatched) {
      return;
    }
    gd._mxWatched = true;
    // A new figure from a callback replaces the layout: theme it again.
    gd.on("plotly_afterplot", function () {
      apply(gd);
      fit(gd);
    });
    apply(gd);
    fit(gd);
  }

  function scan() {
    document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
      if (gd.on) {
        watch(gd);
      }
    });
  }

  function retheme() {
    document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
      gd._mxTheme = null;
      apply(gd);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
    new MutationObserver(retheme).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-mx-theme"],
    });
    scan();
  });
})();
