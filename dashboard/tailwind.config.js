/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', '"IBM Plex Mono"', '"Consolas"', '"Courier New"', "monospace"],
      },
      colors: {
        // Bloomberg-style palette
        term: {
          bg:      "#0a0a0a",   // near-black background
          panel:   "#111111",   // panel background
          border:  "#1f1f1f",   // panel border
          rule:    "#2a2a2a",   // gridlines / rules
          amber:   "#ffaa00",   // primary accent (Bloomberg orange/amber)
          amberDim:"#995f00",
          green:   "#00d97e",   // up / positive
          red:     "#ff3344",   // down / negative
          cyan:    "#00b8d4",   // info
          text:    "#e8e8e8",   // primary text
          muted:   "#888888",   // secondary text
          dim:     "#555555",   // tertiary text
        },
      },
    },
  },
  plugins: [],
}
