// 앱 아이콘·스플래시 원본 이미지를 만든다 (assets/*.png).
//   node assets/make-icons.js            → 원본 PNG 5장 생성 (Playwright 필요)
//   npx @capacitor/assets generate --android   → 안드로이드 해상도별 아이콘으로 변환
const { chromium } = require("playwright");
const path = require("path");

const out = __dirname;
const grad = "linear-gradient(135deg,#ff4f73,#ff8a5c)";
const page = (w, h, body, bg) =>
  `<html><body style="margin:0;width:${w}px;height:${h}px;background:${bg};display:grid;place-items:center;overflow:hidden">${body}</body></html>`;
// 흰색 발바닥 (큰 패드 1개 + 발가락 4개)
const paw = s => `<svg width="${s}" height="${s}" viewBox="0 0 100 100" fill="#fff">
  <ellipse cx="50" cy="66" rx="24" ry="19"/>
  <ellipse cx="19" cy="45" rx="9" ry="12" transform="rotate(-25 19 45)"/>
  <ellipse cx="38" cy="27" rx="9.5" ry="13" transform="rotate(-8 38 27)"/>
  <ellipse cx="62" cy="27" rx="9.5" ry="13" transform="rotate(8 62 27)"/>
  <ellipse cx="81" cy="45" rx="9" ry="12" transform="rotate(25 81 45)"/>
</svg>`;

(async () => {
  const b = await chromium.launch();
  const shot = async (name, w, h, html, transparent = false) => {
    const p = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
    await p.setContent(html);
    await p.screenshot({ path: path.join(out, name), omitBackground: transparent });
    await p.close();
  };
  await shot("icon-only.png", 1024, 1024, page(1024, 1024, paw(600), grad));
  await shot("icon-background.png", 1024, 1024, page(1024, 1024, "", grad));
  await shot("icon-foreground.png", 1024, 1024, page(1024, 1024, paw(400), "transparent"), true);
  const logo = `<div style="width:520px;height:520px;border-radius:160px;background:${grad};display:grid;place-items:center">${paw(320)}</div>`;
  await shot("splash.png", 2732, 2732, page(2732, 2732, logo, "#ffffff"));
  await shot("splash-dark.png", 2732, 2732, page(2732, 2732, logo, "#1b1a21"));
  await b.close();
})();
