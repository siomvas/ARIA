/* ARIA page: the results explorer (Tables 1 to 3 of the paper: 9 architectures x 3 starting weights x 3 aggregation
   methods on Fed-ISIC and two Dirichlet splits of OrganAMNIST) and the BibTeX copy. Every number is the paper's. */
(function () {
  "use strict";
  var NF = new Intl.NumberFormat("en", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  var ARCH = [
    { n: "ResNet-18", s: "11.7M, BN" }, { n: "NF-ResNet-50", s: "25.6M, SWS" }, { n: "ResNet-50", s: "25.6M, BN" },
    { n: "WRN-50-2", s: "68.9M, BN" }, { n: "DenseNet-121", s: "8M, BN" }, { n: "Swin-T", s: "28.3M, LN" },
    { n: "EfficientNetV2-S", s: "21.5M, BN" }, { n: "ViT-B/16", s: "86.6M, LN" }, { n: "ConvNeXt-S", s: "50.2M, LN" }
  ];
  var INIT = ["Random", "ImageNet", "DINO"], AGG = ["FedAvg", "FedOpt", "SCAFFOLD"];
  var D = {
    fedisic: { name: "Fed-ISIC", metric: "balanced accuracy", ssl: "DINO on Skin-SSL", rows: [
      [51.65, 46.70, 52.45, 65.87, 67.55, 68.66, 66.57, 62.36, 66.87],
      [55.93, 56.25, 59.64, 71.88, 68.75, 71.53, 67.83, 67.92, 70.11],
      [49.11, 46.91, 48.13, 67.97, 66.16, 68.48, 65.16, 66.46, 66.34],
      [50.53, 50.12, 51.03, 69.54, 67.68, 70.34, 65.56, 64.22, 66.66],
      [49.42, 45.95, 52.79, 67.34, 68.03, 68.52, 66.28, 64.94, 67.38],
      [45.73, 44.13, 45.00, 71.19, 71.81, 73.13, 72.13, 71.40, 73.77],
      [46.59, 46.59, 47.51, 70.00, 71.48, 73.18, 57.99, 59.74, 64.98],
      [47.84, 49.52, 48.44, 65.86, 65.18, 68.09, 71.06, 71.52, 69.49],
      [48.10, 49.93, 48.56, 75.08, 73.40, 74.28, 72.07, 73.57, 74.56]] },
    organ01: { name: "OrganAMNIST, α = 0.1", metric: "accuracy", ssl: "DINO on Abdomen-SSL", rows: [
      [88.80, 90.76, 89.16, 94.02, 94.78, 94.33, 83.54, 87.89, 84.76],
      [71.60, 78.84, 73.80, 94.39, 95.26, 95.20, 84.58, 87.93, 86.92],
      [83.32, 86.60, 84.82, 91.98, 92.98, 92.32, 81.33, 85.69, 81.49],
      [84.52, 85.58, 83.82, 90.56, 91.71, 90.40, 79.98, 85.02, 77.09],
      [86.01, 89.12, 85.06, 94.72, 95.10, 94.68, 85.26, 89.21, 84.94],
      [83.03, 85.17, 83.16, 95.64, 95.83, 95.83, 83.40, 86.40, 84.80],
      [88.80, 91.46, 89.19, 94.00, 94.26, 93.46, 61.19, 67.54, 56.20],
      [83.14, 83.52, 83.85, 95.30, 95.96, 96.01, 81.34, 83.76, 81.99],
      [53.76, 56.07, 55.34, 94.12, 94.92, 94.84, 87.31, 89.68, 87.64]] },
    organ100: { name: "OrganAMNIST, α = 100", metric: "accuracy", ssl: "DINO on Abdomen-SSL", rows: [
      [93.80, 94.30, 93.97, 96.05, 96.38, 95.99, 92.06, 93.47, 92.14],
      [84.28, 88.09, 84.40, 95.50, 95.64, 95.60, 92.08, 92.74, 92.09],
      [93.39, 94.00, 93.54, 94.98, 95.56, 95.34, 92.80, 93.51, 92.69],
      [93.74, 93.99, 93.72, 94.70, 95.42, 94.76, 92.24, 93.10, 92.52],
      [93.95, 94.28, 93.66, 96.53, 97.00, 96.66, 93.40, 94.11, 93.38],
      [90.64, 90.89, 90.27, 96.61, 96.82, 96.60, 89.66, 90.86, 89.68],
      [94.84, 95.13, 94.96, 96.22, 96.48, 96.28, 89.18, 92.03, 89.02],
      [86.42, 86.34, 86.54, 96.12, 96.30, 96.25, 86.67, 87.61, 86.94],
      [84.56, 87.29, 78.48, 96.30, 96.24, 96.18, 92.15, 92.87, 92.24]] }
  };
  /* a light ramp: the ink stays at 5.4:1 or more on the darkest cell */
  var STOPS = [[0, [238, 242, 253]], [0.5, [179, 196, 246]], [1, [95, 132, 236]]];
  function ramp(t) {
    t = Math.max(0, Math.min(1, t));
    for (var i = 1; i < STOPS.length; i++) {
      if (t <= STOPS[i][0]) {
        var a = STOPS[i - 1], b = STOPS[i], u = (t - a[0]) / (b[0] - a[0]);
        return "rgb(" + a[1].map(function (c, k) { return Math.round(c + (b[1][k] - c) * u); }).join(",") + ")";
      }
    }
    return "rgb(95,132,236)";
  }
  function ordinal(n) { var s = ["th", "st", "nd", "rd"], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); }

  var grid = document.getElementById("ar-grid"), open = document.getElementById("ar-open");
  var title = document.getElementById("ar-title"), status = document.getElementById("ar-status");
  function say(t) { if (status) { status.textContent = ""; setTimeout(function () { status.textContent = t; }, 60); } }
  function setParams(o) { try { var u = new URL(location.href); for (var k in o) u.searchParams.set(k, o[k]); history.replaceState(null, "", u); } catch (e) {} }

  if (grid && open) {
    var q = new URLSearchParams(location.search);
    var state = { set: D[q.get("data")] ? q.get("data") : "fedisic", r: -1, c: -1 };
    var PHONE = window.matchMedia("(max-width: 639px)");
    var m = /^(\d)-(\d)$/.exec(q.get("run") || "");
    if (m) { state.r = +m[1]; state.c = +m[2]; }
    var tabs = document.querySelectorAll(".ar-tabs button");

    var stats = function (rows) {
      var all = [], best = { v: -1, r: 0, c: 0 };
      rows.forEach(function (row, r) { row.forEach(function (v, c) { all.push(v); if (v > best.v) best = { v: v, r: r, c: c }; }); });
      return { lo: Math.min.apply(null, all), hi: Math.max.apply(null, all), best: best, sorted: all.slice().sort(function (a, b) { return b - a; }) };
    };

    var build = function () {
      var S = D[state.set], st = stats(S.rows), wide = window.innerWidth >= 1280, phone = PHONE.matches;
      if (state.r < 0) { state.r = st.best.r; state.c = st.best.c; }
      var g0 = Math.floor(state.c / 3), cols = phone ? [g0 * 3, g0 * 3 + 1, g0 * 3 + 2] : [0, 1, 2, 3, 4, 5, 6, 7, 8];
      grid.innerHTML = "";
      var head1 = document.createElement("div"); head1.setAttribute("role", "row"); head1.style.display = "contents";
      var corner = document.createElement("span"); corner.setAttribute("role", "columnheader"); corner.style.gridRow = "1 / span 2"; corner.style.gridColumn = "1";
      head1.appendChild(corner);
      if (phone) {
        var it = document.createElement("div"); it.className = "ar-itabs"; it.setAttribute("role", "group"); it.setAttribute("aria-label", "Starting weights");
        INIT.forEach(function (name, g) {
          var b = document.createElement("button"); b.type = "button"; b.textContent = name;
          b.setAttribute("aria-pressed", g === g0 ? "true" : "false");
          b.addEventListener("click", function () { state.c = g * 3 + (state.c % 3); build(); pick(state.r, state.c, true); });
          it.appendChild(b);
        });
        head1.appendChild(it);
      } else {
        INIT.forEach(function (name, g) {
          var h = document.createElement("span"); h.className = "ar-grid__g"; h.setAttribute("role", "columnheader");
          h.style.gridColumn = (2 + g * 3) + " / span 3"; h.style.marginLeft = g ? "6px" : "0";
          h.textContent = g === 2 ? S.ssl : name; h.title = g === 2 ? S.ssl : name;
          head1.appendChild(h);
        });
      }
      grid.appendChild(head1);
      var head2 = document.createElement("div"); head2.setAttribute("role", "row"); head2.style.display = "contents";
      cols.forEach(function (c, i) {
        var a = document.createElement("span"); a.className = "ar-grid__a"; a.setAttribute("role", "columnheader");
        a.style.gridColumn = String(2 + i); if (!phone && c % 3 === 0 && c) a.style.marginLeft = "6px";
        a.textContent = (wide || phone) ? AGG[c % 3] : ["Avg", "Opt", "SCAF"][c % 3]; a.title = AGG[c % 3];
        head2.appendChild(a);
      });
      grid.appendChild(head2);
      S.rows.forEach(function (row, r) {
        var tr = document.createElement("div"); tr.setAttribute("role", "row"); tr.style.display = "contents";
        var rh = document.createElement("span"); rh.className = "ar-grid__r"; rh.setAttribute("role", "rowheader");
        rh.innerHTML = '<span translate="no">' + ARCH[r].n + "</span><small>" + ARCH[r].s + "</small>";
        tr.appendChild(rh);
        cols.forEach(function (c) {
          var v = row[c], t = (v - st.lo) / (st.hi - st.lo), b = document.createElement("button");
          b.type = "button"; b.className = "ar-cell" + (!phone && c % 3 === 0 && c ? " is-g" : "") + (r === st.best.r && c === st.best.c ? " is-best" : "");
          b.setAttribute("role", "gridcell"); b.dataset.r = r; b.dataset.c = c;
          b.dataset.fill = ramp(t);
          b.innerHTML = "<span>" + NF.format(v) + "</span>";
          b.setAttribute("aria-label", ARCH[r].n + ", " + (Math.floor(c / 3) === 2 ? S.ssl : INIT[Math.floor(c / 3)]) + ", " + AGG[c % 3] + ": " + NF.format(v));
          b.setAttribute("aria-selected", r === state.r && c === state.c ? "true" : "false");
          b.tabIndex = r === state.r && c === state.c ? 0 : -1;
          b.addEventListener("click", function () { pick(r, c, true); });
          tr.appendChild(b);
        });
        grid.appendChild(tr);
      });
      paint();
      detail();
    };

    /* the shades of the 81 runs: one picture under the grid, a rectangle under each cell */
    var paint = function () {
      var old = grid.querySelector("svg.ar-grid__bg"); if (old) old.remove();
      var w = grid.clientWidth, h = grid.clientHeight, ns = "http://www.w3.org/2000/svg";
      if (!w || !h) return;
      var svg = document.createElementNS(ns, "svg");
      svg.setAttribute("class", "ar-grid__bg"); svg.setAttribute("aria-hidden", "true");
      svg.setAttribute("width", w); svg.setAttribute("height", h); svg.setAttribute("viewBox", "0 0 " + w + " " + h);
      grid.querySelectorAll(".ar-cell").forEach(function (b) {
        var r = document.createElementNS(ns, "rect");
        r.setAttribute("x", b.offsetLeft); r.setAttribute("y", b.offsetTop);
        r.setAttribute("width", b.offsetWidth); r.setAttribute("height", b.offsetHeight);
        r.setAttribute("rx", 4); r.setAttribute("fill", b.dataset.fill);
        svg.appendChild(r);
      });
      grid.insertBefore(svg, grid.firstChild);
    };

    var detail = function () {
      var S = D[state.set], st = stats(S.rows), r = state.r, c = state.c, v = S.rows[r][c];
      var g = Math.floor(c / 3), initName = g === 2 ? S.ssl : INIT[g] + (g === 1 ? " pre-training" : " weights");
      var rank = st.sorted.indexOf(v) + 1;
      var others = [0, 1, 2].filter(function (k) { return k !== c % 3; }).map(function (k) { return AGG[k] + " " + NF.format(S.rows[r][g * 3 + k]); });
      var rowBest = S.rows[r].indexOf(Math.max.apply(null, S.rows[r]));
      title.textContent = ARCH[r].n + ", " + (g === 2 ? "DINO" : INIT[g]) + ", " + AGG[c % 3];
      open.innerHTML =
        '<div><div class="ar-open__v">' + NF.format(v) + '</div><div class="ar-open__m">' + S.metric + " on " + S.name + "</div></div>" +
        "<dl><dt>Rank</dt><dd>" + ordinal(rank) + " of 81</dd>" +
        "<dt>Model</dt><dd>" + ARCH[r].n + ", " + ARCH[r].s + "</dd>" +
        "<dt>Start</dt><dd>" + initName + "</dd>" +
        "<dt>Same start</dt><dd>" + others.join(", ") + "</dd>" +
        "<dt>Its best</dt><dd>" + (Math.floor(rowBest / 3) === 2 ? "DINO" : INIT[Math.floor(rowBest / 3)]) + ", " + AGG[rowBest % 3] + ", " + NF.format(S.rows[r][rowBest]) + "</dd></dl>" +
        '<div class="ar-bars">' + bars(S, st, r, c) + "</div>" +
        '<div class="ar-scale"><i aria-hidden="true"></i><div><span>' + NF.format(st.lo) + "</span><span>" + NF.format(st.hi) + "</span></div></div>";
    };

    var bars = function (S, st, r, c) {
      var w = 320, h = 132, top = 18, base = 104, gw = (w - 2 * 12) / 3, bw = 22, row = S.rows[r];
      var y = function (v) { return base - (v - st.lo) / (st.hi - st.lo) * (base - top - 8) - 8; };
      var out = '<svg viewBox="0 0 ' + w + " " + h + '" role="img" aria-label="' + ARCH[r].n + " on " + S.name + ", all nine runs: " + row.map(function (v, k) { return (Math.floor(k / 3) === 2 ? "DINO" : INIT[Math.floor(k / 3)]) + " " + AGG[k % 3] + " " + NF.format(v); }).join(", ") + '">';
      out += '<line x1="0" x2="' + w + '" y1="' + base + '" y2="' + base + '" stroke="rgba(17,17,17,.14)"/>';
      for (var g = 0; g < 3; g++) {
        var gx = g * (gw + 12);
        for (var k = 0; k < 3; k++) {
          var i = g * 3 + k, x = gx + 8 + k * (bw + 10), yy = y(row[i]), on = i === c;
          out += '<rect x="' + x + '" y="' + yy.toFixed(1) + '" width="' + bw + '" height="' + (base - yy).toFixed(1) + '" rx="3" fill="' + (on ? "#1a4be0" : "#b3c4f6") + '"/>';
        }
        out += '<text x="' + (gx + 8) + '" y="' + (base + 20) + '" font-size="12" fill="#555">' + (g === 2 ? "DINO" : INIT[g]) + "</text>";
      }
      return out + "</svg>";
    };

    var pick = function (r, c, announce) {
      state.r = r; state.c = c;
      grid.querySelectorAll(".ar-cell").forEach(function (b) {
        var on = +b.dataset.r === r && +b.dataset.c === c;
        b.setAttribute("aria-selected", on ? "true" : "false"); b.tabIndex = on ? 0 : -1;
      });
      detail();
      setParams({ data: state.set, run: r + "-" + c });
      if (announce) { var S = D[state.set]; say(ARCH[r].n + ", " + AGG[c % 3] + ": " + NF.format(S.rows[r][c]) + " " + S.metric + "."); }
    };

    grid.addEventListener("keydown", function (e) {
      var k = e.key, r = state.r, c = state.c;
      var lo = PHONE.matches ? Math.floor(c / 3) * 3 : 0, hi = PHONE.matches ? lo + 2 : 8;
      if (k === "ArrowRight") c = Math.min(hi, c + 1); else if (k === "ArrowLeft") c = Math.max(lo, c - 1);
      else if (k === "ArrowDown") r = Math.min(8, r + 1); else if (k === "ArrowUp") r = Math.max(0, r - 1);
      else if (k === "Home") c = lo; else if (k === "End") c = hi; else return;
      e.preventDefault(); pick(r, c, true);
      var b = grid.querySelector('.ar-cell[data-r="' + r + '"][data-c="' + c + '"]'); if (b) b.focus();
    });
    tabs.forEach(function (t) {
      t.addEventListener("click", function () {
        state.set = t.getAttribute("data-set"); state.r = -1; state.c = -1;
        tabs.forEach(function (o) { o.setAttribute("aria-pressed", o === t ? "true" : "false"); });
        build(); setParams({ data: state.set, run: state.r + "-" + state.c });
        say(D[state.set].name + ": the best run is " + ARCH[state.r].n + ", " + AGG[state.c % 3] + ".");
      });
      t.setAttribute("aria-pressed", t.getAttribute("data-set") === state.set ? "true" : "false");
    });
    build();
    if (PHONE.addEventListener) PHONE.addEventListener("change", function () { build(); });
    if ("ResizeObserver" in window) { var gw = 0, gh = 0; new ResizeObserver(function () { if (grid.clientWidth !== gw || grid.clientHeight !== gh) { gw = grid.clientWidth; gh = grid.clientHeight; paint(); } }).observe(grid); }
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(paint);
  }

  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var src = document.getElementById(btn.getAttribute("data-copy"));
      if (!src || !navigator.clipboard) return;
      navigator.clipboard.writeText(src.textContent).then(function () {
        var was = btn.textContent; btn.textContent = "Copied"; say("BibTeX copied to the clipboard.");
        setTimeout(function () { btn.textContent = was; }, 1600);
      });
    });
  });
})();
