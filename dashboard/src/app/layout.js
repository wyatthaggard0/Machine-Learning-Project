import "./globals.css"

export const metadata = {
  title: "Fraud Detection Dashboard",
  description: "Machine learning model results for IEEE-CIS credit card fraud detection",
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
