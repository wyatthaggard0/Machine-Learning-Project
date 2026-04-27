/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      },
      colors: {
        bg:        "#f7f8fa",
        card:      "#ffffff",
        border:    "#e5e7eb",
        rule:      "#eef0f3",
        ink:       "#1a1f2e",   // primary text
        body:      "#4b5563",   // body text
        muted:     "#6b7280",   // labels
        dim:       "#9ca3af",   // faint
        accent:    "#2563eb",   // primary blue accent
        accentSoft:"#dbeafe",
        good:      "#059669",   // green / safe
        warn:      "#d97706",   // amber / review
        bad:       "#dc2626",   // red / fraud
      },
    },
  },
  plugins: [],
}
