/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans:  ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        serif: ['"Playfair Display"', 'Georgia', 'serif'],
        mono:  ['"JetBrains Mono"', 'Consolas', 'monospace'],
      },
      colors: {
        ink: {
          paper:  "#f5f1e8",   // warm off-white background
          card:   "#fffdf8",   // card surface
          rule:   "#d4cab1",   // hairline borders
          navy:   "#0b2545",   // deep navy — primary text & accents
          ink:    "#13315c",   // mid-navy — body emphasis
          coral:  "#e63946",   // alert / fraud-pushing accent
          sage:   "#588157",   // safe / non-fraud accent
          gold:   "#c9a227",   // highlight / featured
          mute:   "#6c757d",   // secondary text
          dim:    "#a0978a",   // tertiary text
          tint:   "#ebe5d6",   // muted background panel
        },
      },
    },
  },
  plugins: [],
}
