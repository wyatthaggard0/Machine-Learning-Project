/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'sans-serif'],
      },
      colors: {
        // Databox-inspired forest-green palette
        db: {
          bg:     "#0e3a2c",   // page background — deep forest
          panel:  "#1d5742",   // card / panel — slightly lighter forest
          panel2: "#164534",   // alternate panel
          border: "#2a7257",   // soft border
          rule:   "#2a7257",   // dividers
          orange: "#ff6a13",   // primary accent (Databox orange)
          orangeSoft: "rgba(255,106,19,0.18)",
          amber:  "#ffa84a",   // soft accent
          green:  "#22c773",   // up / positive
          greenSoft: "rgba(34,199,115,0.2)",
          red:    "#ff5d5d",   // down / negative
          redSoft: "rgba(255,93,93,0.2)",
          text:   "#ffffff",   // primary text
          muted:  "#bfd6cb",   // secondary text
          dim:    "#80a194",   // tertiary text
        },
      },
    },
  },
  plugins: [],
}
