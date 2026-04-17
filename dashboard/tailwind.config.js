/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          green: "#2ecc71",
          red:   "#e74c3c",
          blue:  "#3498db",
          dark:  "#1a1a2e",
          card:  "#16213e",
          muted: "#0f3460",
        },
      },
    },
  },
  plugins: [],
}
